/* CoreMIDI shim for the NAM example (mirrors audio_shim.c: one C file, a
 * narrow extern ABI, no C types leaking into Zig beyond what midi.zig
 * declares). Source enumeration + an input port that forwards each raw
 * MIDI packet to a Zig callback; parsing happens on the Zig side where it
 * is unit-testable. Non-Apple builds compile the same ABI as stubs and
 * the Zig side treats create()==NULL as "no MIDI backend". */

#ifdef __APPLE__

#include <CoreFoundation/CoreFoundation.h>
#include <CoreMIDI/CoreMIDI.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* `source` is the connection index passed at MIDIPortConnectSource time, so
 * the Zig side can keep per-source parser state (multi-packet SysEx from
 * one device must not corrupt another device's byte stream). */
typedef void (*nam_midi_callback)(void* user, unsigned int source, const unsigned char* bytes, unsigned int len);

typedef struct {
    MIDIClientRef client;
    MIDIPortRef port;
    /* Written by the UI thread (start/stop), read by CoreMIDI's thread:
     * release/acquire so the proc never sees a half-published pair. */
    _Atomic(nam_midi_callback) callback;
    void* user;
    int client_ready;
    int port_ready;
} nam_midi;

void nam_midi_stop(nam_midi* midi);

/* The legacy packet-list read proc delivers plain MIDI 1.0 bytes; its
 * macOS-11 replacement (MIDIInputPortCreateWithProtocol) needs an ObjC
 * block and delivers UMP words. Deprecated but fully functional, and the
 * simplest ABI for a C-only shim. It runs on CoreMIDI's own high-priority
 * thread — never on the audio callback. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

static void nam_midi_read_proc(const MIDIPacketList* list, void* ref, void* conn_ref) {
    nam_midi* midi = (nam_midi*)ref;
    /* Snapshot the pair once: a concurrent stop() must not turn the
     * per-packet call into a jump through NULL mid-list, and `user` must
     * be the value published WITH this callback (the acquire pairs with
     * start()'s release store, which is placed after the user write). */
    nam_midi_callback cb = atomic_load_explicit(&midi->callback, memory_order_acquire);
    if (cb == NULL) return;
    void* user = midi->user;
    const MIDIPacket* packet = &list->packet[0];
    for (UInt32 i = 0; i < list->numPackets; i++) {
        cb(user, (unsigned int)(uintptr_t)conn_ref, packet->data, (unsigned int)packet->length);
        packet = MIDIPacketNext(packet);
    }
}

/* CoreMIDI publishes device-list changes as notifications on the runloop
 * of the thread that first called into it — which a CLI never spins, so
 * MIDIGetNumberOfSources() would answer from a snapshot frozen at client
 * creation (verified live: a source published after startup stayed
 * invisible). Draining the runloop before enumerating keeps the snapshot
 * current. Must run on the client-creating thread — here that's the UI
 * thread, which owns every enumeration call. */
static void nam_midi_pump_runloop(void) {
    while (CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0, true) == kCFRunLoopRunHandledSource) {}
}

nam_midi* nam_midi_create(void) {
    nam_midi* midi = (nam_midi*)calloc(1, sizeof(nam_midi));
    if (midi == NULL) return NULL;
    /* No notify proc: hot-plug is handled by the caller polling
     * nam_midi_source_count() (which pumps the runloop) instead. */
    if (MIDIClientCreate(CFSTR("fucina-nam"), NULL, NULL, &midi->client) != noErr) {
        free(midi);
        return NULL;
    }
    midi->client_ready = 1;
    return midi;
}

void nam_midi_destroy(nam_midi* midi) {
    if (midi == NULL) return;
    nam_midi_stop(midi);
    if (midi->client_ready) MIDIClientDispose(midi->client);
    free(midi);
}

int nam_midi_source_count(nam_midi* midi) {
    (void)midi;
    nam_midi_pump_runloop();
    return (int)MIDIGetNumberOfSources();
}

/* Order-sensitive hash of (count, per-source unique IDs): the hot-plug
 * rescan compares this instead of the bare count, so unplugging one
 * controller and plugging another inside a single poll window still
 * registers as a change (a count compare would see n -> n). */
int64_t nam_midi_sources_signature(nam_midi* midi) {
    (void)midi;
    nam_midi_pump_runloop();
    ItemCount count = MIDIGetNumberOfSources();
    int64_t hash = (int64_t)count;
    for (ItemCount i = 0; i < count; i++) {
        SInt32 uid = 0;
        MIDIObjectGetIntegerProperty(MIDIGetSource(i), kMIDIPropertyUniqueID, &uid);
        hash = hash * 1000003 + (int64_t)uid;
    }
    return hash;
}

/* Copies up to `cap` NUL-terminated display names into name_buf + i*name_cap;
 * returns the total source count. */
int nam_midi_list_sources(nam_midi* midi, char* name_buf, int name_cap, int cap) {
    (void)midi;
    nam_midi_pump_runloop();
    int count = (int)MIDIGetNumberOfSources();
    for (int i = 0; i < count && i < cap; i++) {
        char* dst = name_buf + (size_t)i * (size_t)name_cap;
        dst[0] = 0;
        CFStringRef name = NULL;
        if (MIDIObjectGetStringProperty(MIDIGetSource((ItemCount)i), kMIDIPropertyDisplayName, &name) == noErr && name != NULL) {
            /* On conversion failure the buffer holds partial bytes with no
             * NUL — reset it so the fallback below fires. */
            if (!CFStringGetCString(name, dst, name_cap, kCFStringEncodingUTF8)) dst[0] = 0;
            CFRelease(name);
        }
        if (dst[0] == 0) snprintf(dst, (size_t)name_cap, "midi source %d", i);
    }
    return count;
}

/* source_index = -1 connects every source (omni). Returns the number of
 * connected sources (0 is fine: nothing plugged in yet), or negative on
 * error. Restarting on an already-open port is allowed (hot-plug rescan). */
int nam_midi_start(nam_midi* midi, int source_index, nam_midi_callback callback, void* user) {
    if (midi == NULL || !midi->client_ready) return -1;
    nam_midi_stop(midi);
    midi->user = user;
    if (MIDIInputPortCreate(midi->client, CFSTR("fucina-nam-in"), nam_midi_read_proc, midi, &midi->port) != noErr) {
        return -2;
    }
    midi->port_ready = 1;
    nam_midi_pump_runloop();
    int count = (int)MIDIGetNumberOfSources();
    int connected = 0;
    if (source_index >= 0) {
        if (source_index >= count) {
            nam_midi_stop(midi);
            return -3;
        }
        if (MIDIPortConnectSource(midi->port, MIDIGetSource((ItemCount)source_index), (void*)(uintptr_t)source_index) == noErr) connected++;
    } else {
        for (int i = 0; i < count; i++) {
            if (MIDIPortConnectSource(midi->port, MIDIGetSource((ItemCount)i), (void*)(uintptr_t)i) == noErr) connected++;
        }
    }
    /* Publish the callback last (release): a packet racing the connect
     * loop is dropped instead of dispatched half-initialized, and the
     * `user` store above is ordered before the publication. */
    atomic_store_explicit(&midi->callback, callback, memory_order_release);
    return connected;
}

void nam_midi_stop(nam_midi* midi) {
    if (midi == NULL) return;
    /* Unpublish first; the proc snapshots the pointer once per packet
     * list, so an already-in-flight delivery still lands in the caller's
     * stream (which outlives the port), never through NULL. */
    atomic_store_explicit(&midi->callback, NULL, memory_order_release);
    if (midi->port_ready) {
        /* Conventional CoreMIDI teardown (RtMidi/JUCE do the same): Apple
         * documents no quiescence guarantee for MIDIPortDispose, which is
         * why the callback is unpublished first and the receiving stream
         * must outlive the port. */
        MIDIPortDispose(midi->port);
        midi->port_ready = 0;
    }
}

#pragma clang diagnostic pop

#else /* !__APPLE__ */

#include <stdint.h>

typedef void (*nam_midi_callback)(void* user, unsigned int source, const unsigned char* bytes, unsigned int len);
typedef struct nam_midi nam_midi;

nam_midi* nam_midi_create(void) { return 0; }
void nam_midi_destroy(nam_midi* midi) { (void)midi; }
int nam_midi_source_count(nam_midi* midi) { (void)midi; return 0; }
int64_t nam_midi_sources_signature(nam_midi* midi) { (void)midi; return 0; }
int nam_midi_list_sources(nam_midi* midi, char* name_buf, int name_cap, int cap) {
    (void)midi; (void)name_buf; (void)name_cap; (void)cap;
    return 0;
}
int nam_midi_start(nam_midi* midi, int source_index, nam_midi_callback callback, void* user) {
    (void)midi; (void)source_index; (void)callback; (void)user;
    return -1;
}
void nam_midi_stop(nam_midi* midi) { (void)midi; }

#endif
