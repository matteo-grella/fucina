/* Single-TU miniaudio build + a narrow extern-C ABI for the NAM example
 * (mirrors the repo's Metal shim.m pattern: one C file, a handful of
 * extern functions, no C types leaking into Zig beyond what's declared in
 * audio.zig). Duplex (capture -> user callback -> playback) with f32 mono
 * frames at a fixed period size; same-device duplex is the recommended
 * configuration: one clock, so no drift. Split in/out devices run on
 * independent clocks with no drift correction in miniaudio, so latency
 * creeps and a periodic click is possible. */

#include "miniaudio_config.h"
#define MINIAUDIO_IMPLEMENTATION
#include "third_party/miniaudio.h"

#include <string.h>

typedef void (*nam_audio_callback)(void* user, float* output, const float* input, unsigned int frame_count);

typedef struct {
    ma_context context;
    ma_device device;
    nam_audio_callback callback;
    void* user;
    int context_ready;
    int device_ready;
} nam_audio;

static void nam_device_callback(ma_device* device, void* output, const void* input, ma_uint32 frame_count) {
    nam_audio* audio = (nam_audio*)device->pUserData;
    /* output is NULL for capture-only streams; the Zig side handles it. */
    if (audio->callback != NULL) {
        audio->callback(audio->user, (float*)output, (const float*)input, frame_count);
    } else if (output != NULL) {
        memset(output, 0, (size_t)frame_count * sizeof(float));
    }
}

nam_audio* nam_audio_create(void) {
    nam_audio* audio = (nam_audio*)MA_MALLOC(sizeof(nam_audio));
    if (audio == NULL) return NULL;
    memset(audio, 0, sizeof(*audio));
    ma_context_config context_config = ma_context_config_init();
    /* Realtime priority for any thread miniaudio itself spawns (CoreAudio
     * callbacks already run on the HAL's realtime thread; this covers the
     * rest — recommended in miniaudio's own latency guidance). */
    context_config.threadPriority = ma_thread_priority_realtime;
    if (ma_context_init(NULL, 0, &context_config, &audio->context) != MA_SUCCESS) {
        MA_FREE(audio);
        return NULL;
    }
    audio->context_ready = 1;
    return audio;
}

void nam_audio_destroy(nam_audio* audio) {
    if (audio == NULL) return;
    if (audio->device_ready) ma_device_uninit(&audio->device);
    if (audio->context_ready) ma_context_uninit(&audio->context);
    MA_FREE(audio);
}

/* Writes up to `cap` device descriptions; returns the total count of the
 * requested kind (kind 0 = playback, 1 = capture). Each name is copied
 * NUL-terminated into name_buf + i*name_cap. */
int nam_audio_list_devices(nam_audio* audio, int kind, char* name_buf, int name_cap, unsigned char* default_flags, int cap) {
    ma_device_info* playback_infos;
    ma_uint32 playback_count;
    ma_device_info* capture_infos;
    ma_uint32 capture_count;
    if (ma_context_get_devices(&audio->context, &playback_infos, &playback_count, &capture_infos, &capture_count) != MA_SUCCESS) {
        return -1;
    }
    ma_device_info* infos = kind == 0 ? playback_infos : capture_infos;
    ma_uint32 count = kind == 0 ? playback_count : capture_count;
    ma_uint32 n = count < (ma_uint32)cap ? count : (ma_uint32)cap;
    for (ma_uint32 i = 0; i < n; i += 1) {
        strncpy(name_buf + (size_t)i * (size_t)name_cap, infos[i].name, (size_t)name_cap - 1);
        name_buf[(size_t)i * (size_t)name_cap + (size_t)name_cap - 1] = '\0';
        default_flags[i] = infos[i].isDefault ? 1 : 0;
    }
    return (int)count;
}

/* Opens a full-duplex mono f32 stream. capture_index / playback_index are
 * indices into the respective enumeration order, or -1 for the system
 * default. Returns 0 on success. */
int nam_audio_start(
    nam_audio* audio,
    int capture_index,
    int playback_index,
    unsigned int sample_rate,
    unsigned int period_frames,
    nam_audio_callback callback,
    void* user) {
    if (audio->device_ready) return -1;

    ma_device_info* playback_infos;
    ma_uint32 playback_count;
    ma_device_info* capture_infos;
    ma_uint32 capture_count;
    if (ma_context_get_devices(&audio->context, &playback_infos, &playback_count, &capture_infos, &capture_count) != MA_SUCCESS) {
        return -2;
    }
    if (capture_index >= (int)capture_count || playback_index >= (int)playback_count) return -3;

    audio->callback = callback;
    audio->user = user;

    ma_device_config config = ma_device_config_init(ma_device_type_duplex);
    if (capture_index >= 0) config.capture.pDeviceID = &capture_infos[capture_index].id;
    if (playback_index >= 0) config.playback.pDeviceID = &playback_infos[playback_index].id;
    config.capture.format = ma_format_f32;
    config.capture.channels = 1;
    config.playback.format = ma_format_f32;
    config.playback.channels = 1;
    config.sampleRate = sample_rate;
    config.periodSizeInFrames = period_frames;
    config.performanceProfile = ma_performance_profile_low_latency;
    /* Deliver whatever the device gives instead of re-buffering to a fixed
     * size: removes one intermediary buffer layer (the engines accept any
     * frame count up to their reset cap). */
    config.noFixedSizedCallback = MA_TRUE;
    /* Skip miniaudio's output pre-zeroing and f32 clip pass: the callback
     * always writes every frame, and clipping is monitored at our gain
     * stage (CLIP! indicator) rather than silently clamped. */
    config.noPreSilencedOutputBuffer = MA_TRUE;
    config.noClip = MA_TRUE;
#ifdef __APPLE__
    /* Retune the device's nominal rate to ours at the OS level. Without
     * this, a 44.1 kHz-native device gets a hidden resampler whose rate
     * error ACCUMULATES in the duplex ring — measured: latency growing
     * ~1100 samples/s until the ring wraps. This is what pro audio apps
     * do; it changes the device rate system-wide for the session. */
    config.coreaudio.allowNominalSampleRateChange = MA_TRUE;
#endif
    config.dataCallback = nam_device_callback;
    config.pUserData = audio;

    if (ma_device_init(&audio->context, &config, &audio->device) != MA_SUCCESS) return -4;
    audio->device_ready = 1;
    if (ma_device_start(&audio->device) != MA_SUCCESS) {
        ma_device_uninit(&audio->device);
        audio->device_ready = 0;
        return -5;
    }
    return 0;
}

/* Capture-only stream (input probing): same contract as nam_audio_start
 * but no playback side; the callback receives output == NULL. */
int nam_audio_start_capture(
    nam_audio* audio,
    int capture_index,
    unsigned int sample_rate,
    unsigned int period_frames,
    nam_audio_callback callback,
    void* user) {
    if (audio->device_ready) return -1;

    ma_device_info* playback_infos;
    ma_uint32 playback_count;
    ma_device_info* capture_infos;
    ma_uint32 capture_count;
    if (ma_context_get_devices(&audio->context, &playback_infos, &playback_count, &capture_infos, &capture_count) != MA_SUCCESS) {
        return -2;
    }
    if (capture_index >= (int)capture_count) return -3;

    audio->callback = callback;
    audio->user = user;

    ma_device_config config = ma_device_config_init(ma_device_type_capture);
    if (capture_index >= 0) config.capture.pDeviceID = &capture_infos[capture_index].id;
    config.capture.format = ma_format_f32;
    config.capture.channels = 1;
    config.sampleRate = sample_rate;
    config.periodSizeInFrames = period_frames;
    config.dataCallback = nam_device_callback;
    config.pUserData = audio;

    if (ma_device_init(&audio->context, &config, &audio->device) != MA_SUCCESS) return -4;
    audio->device_ready = 1;
    if (ma_device_start(&audio->device) != MA_SUCCESS) {
        ma_device_uninit(&audio->device);
        audio->device_ready = 0;
        return -5;
    }
    return 0;
}

void nam_audio_stop(nam_audio* audio) {
    if (!audio->device_ready) return;
    ma_device_uninit(&audio->device);
    audio->device_ready = 0;
    audio->callback = NULL;
}

unsigned int nam_audio_actual_sample_rate(nam_audio* audio) {
    return audio->device_ready ? audio->device.sampleRate : 0;
}

/* The device's native rate on the given side (kind 0 = playback,
 * 1 = capture). When this differs from the stream rate, miniaudio is
 * resampling internally — extra latency the user should know about. */
unsigned int nam_audio_internal_sample_rate(nam_audio* audio, int kind) {
    if (!audio->device_ready) return 0;
    return kind == 0 ? audio->device.playback.internalSampleRate : audio->device.capture.internalSampleRate;
}

/* The device-side period actually negotiated (can exceed the requested
 * one); the duplex ring pre-seeks 2x the CAPTURE internal period, so this
 * is the honest latency input. */
unsigned int nam_audio_internal_period_frames(nam_audio* audio, int kind) {
    if (!audio->device_ready) return 0;
    return kind == 0 ? audio->device.playback.internalPeriodSizeInFrames : audio->device.capture.internalPeriodSizeInFrames;
}

#ifdef __APPLE__
static unsigned int nam_coreaudio_property_u32(AudioObjectID device_id, AudioObjectPropertySelector selector, AudioObjectPropertyScope scope) {
    AudioObjectPropertyAddress address;
    UInt32 value = 0;
    UInt32 size = sizeof(value);
    address.mSelector = selector;
    address.mScope = scope;
    address.mElement = kAudioObjectPropertyElementMain;
    if (AudioObjectGetPropertyData(device_id, &address, 0, NULL, &size, &value) != noErr) return 0;
    return value;
}
#endif

/* Real per-device latency in frames (device latency + safety offset +
 * device buffer), straight from the CoreAudio properties. 0 when unknown
 * (non-macOS or query failure). kind 0 = playback, 1 = capture. */
unsigned int nam_audio_device_latency_frames(nam_audio* audio, int kind) {
#ifdef __APPLE__
    if (!audio->device_ready) return 0;
    AudioObjectID device_id = kind == 0 ? audio->device.coreaudio.deviceObjectIDPlayback : audio->device.coreaudio.deviceObjectIDCapture;
    AudioObjectPropertyScope scope = kind == 0 ? kAudioObjectPropertyScopeOutput : kAudioObjectPropertyScopeInput;
    if (device_id == 0) return 0;
    unsigned int frames = nam_coreaudio_property_u32(device_id, kAudioDevicePropertyLatency, scope)
                        + nam_coreaudio_property_u32(device_id, kAudioDevicePropertySafetyOffset, scope)
                        + nam_coreaudio_property_u32(device_id, kAudioDevicePropertyBufferFrameSize, scope);
    /* Per-stream latency rides on the device's first stream and is NOT
     * included in the device-level properties. */
    {
        AudioObjectPropertyAddress address;
        AudioStreamID streams[8];
        UInt32 size = sizeof(streams);
        address.mSelector = kAudioDevicePropertyStreams;
        address.mScope = scope;
        address.mElement = kAudioObjectPropertyElementMain;
        if (AudioObjectGetPropertyData(device_id, &address, 0, NULL, &size, streams) == noErr && size >= sizeof(AudioStreamID)) {
            frames += nam_coreaudio_property_u32(streams[0], kAudioStreamPropertyLatency, kAudioObjectPropertyScopeGlobal);
        }
    }
    return frames;
#else
    (void)audio;
    (void)kind;
    return 0;
#endif
}

/* Copies the names of the running device pair (capture, playback). */
void nam_audio_running_names(nam_audio* audio, char* capture_name, char* playback_name, int cap) {
    if (!audio->device_ready) {
        if (cap > 0) {
            capture_name[0] = '\0';
            playback_name[0] = '\0';
        }
        return;
    }
    ma_device_get_name(&audio->device, ma_device_type_capture, capture_name, (size_t)cap, NULL);
    ma_device_get_name(&audio->device, ma_device_type_playback, playback_name, (size_t)cap, NULL);
}
