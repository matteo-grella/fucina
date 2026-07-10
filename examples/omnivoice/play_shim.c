/* Playback-only miniaudio shim for the OmniVoice example (--play).
 *
 * NAM's audio_shim.c is the single MINIAUDIO_IMPLEMENTATION TU and its
 * device entry point is duplex-only (capture + playback — opening it would
 * trip the macOS microphone permission for a pure TTS tool), so this file
 * declares its own playback-only device over the SAME vendored miniaudio
 * build: it includes the header in declaration mode and links against the
 * implementation compiled in examples/nam/audio_shim.c (build.zig's
 * configureOmnivoiceAudio adds both TUs). Both TUs take the MA_NO_* config
 * from the shared ../nam/miniaudio_config.h so their view of the header
 * cannot diverge across a future miniaudio version bump.
 *
 * Unlike the NAM duplex shim this one does NOT retune the device's nominal
 * rate (allowNominalSampleRateChange): playback has no drift-accumulation
 * problem, so miniaudio's internal resampler converts the 24 kHz synthesis
 * rate to whatever the device runs natively. */

#include "../nam/miniaudio_config.h"
#include "../nam/third_party/miniaudio.h"

#include <stdlib.h>
#include <string.h>

typedef void (*ov_play_callback)(void* user, float* output, unsigned int frame_count);

typedef struct {
    ma_context context;
    ma_device device;
    ov_play_callback callback;
    void* user;
    int context_ready;
    int device_ready;
} ov_play;

static void ov_play_device_callback(ma_device* device, void* output, const void* input, ma_uint32 frame_count) {
    ov_play* play = (ov_play*)device->pUserData;
    (void)input;
    if (play->callback != NULL) {
        play->callback(play->user, (float*)output, frame_count);
    } else {
        memset(output, 0, (size_t)frame_count * sizeof(float));
    }
}

ov_play* ov_play_create(void) {
    ov_play* play = (ov_play*)malloc(sizeof(ov_play));
    if (play == NULL) return NULL;
    memset(play, 0, sizeof(*play));
    ma_context_config context_config = ma_context_config_init();
    context_config.threadPriority = ma_thread_priority_realtime;
    if (ma_context_init(NULL, 0, &context_config, &play->context) != MA_SUCCESS) {
        free(play);
        return NULL;
    }
    play->context_ready = 1;
    return play;
}

void ov_play_destroy(ov_play* play) {
    if (play == NULL) return;
    if (play->device_ready) ma_device_uninit(&play->device);
    if (play->context_ready) ma_context_uninit(&play->context);
    free(play);
}

/* Writes up to `cap` playback device descriptions; returns the total count.
 * Each name is copied NUL-terminated into name_buf + i*name_cap (the same
 * contract as nam_audio_list_devices, playback side only). */
int ov_play_list_devices(ov_play* play, char* name_buf, int name_cap, unsigned char* default_flags, int cap) {
    ma_device_info* playback_infos;
    ma_uint32 playback_count;
    ma_device_info* capture_infos;
    ma_uint32 capture_count;
    if (ma_context_get_devices(&play->context, &playback_infos, &playback_count, &capture_infos, &capture_count) != MA_SUCCESS) {
        return -1;
    }
    ma_uint32 n = playback_count < (ma_uint32)cap ? playback_count : (ma_uint32)cap;
    for (ma_uint32 i = 0; i < n; i += 1) {
        strncpy(name_buf + (size_t)i * (size_t)name_cap, playback_infos[i].name, (size_t)name_cap - 1);
        name_buf[(size_t)i * (size_t)name_cap + (size_t)name_cap - 1] = '\0';
        default_flags[i] = playback_infos[i].isDefault ? 1 : 0;
    }
    return (int)playback_count;
}

/* Opens a playback-only mono f32 stream. playback_index is an index into
 * the enumeration order, or -1 for the system default. Returns 0 on
 * success (error codes mirror nam_audio_start). */
int ov_play_start(
    ov_play* play,
    int playback_index,
    unsigned int sample_rate,
    unsigned int period_frames,
    ov_play_callback callback,
    void* user) {
    if (play->device_ready) return -1;

    ma_device_info* playback_infos;
    ma_uint32 playback_count;
    ma_device_info* capture_infos;
    ma_uint32 capture_count;
    if (ma_context_get_devices(&play->context, &playback_infos, &playback_count, &capture_infos, &capture_count) != MA_SUCCESS) {
        return -2;
    }
    if (playback_index >= (int)playback_count) return -3;

    play->callback = callback;
    play->user = user;

    ma_device_config config = ma_device_config_init(ma_device_type_playback);
    if (playback_index >= 0) config.playback.pDeviceID = &playback_infos[playback_index].id;
    config.playback.format = ma_format_f32;
    config.playback.channels = 1;
    config.sampleRate = sample_rate;
    config.periodSizeInFrames = period_frames;
    config.dataCallback = ov_play_device_callback;
    config.pUserData = play;

    if (ma_device_init(&play->context, &config, &play->device) != MA_SUCCESS) return -4;
    play->device_ready = 1;
    if (ma_device_start(&play->device) != MA_SUCCESS) {
        ma_device_uninit(&play->device);
        play->device_ready = 0;
        return -5;
    }
    return 0;
}

void ov_play_stop(ov_play* play) {
    if (!play->device_ready) return;
    ma_device_uninit(&play->device);
    play->device_ready = 0;
    play->callback = NULL;
}

/* The playback device's native rate; differing from the stream rate means
 * miniaudio is resampling internally (expected for 24 kHz synthesis). */
unsigned int ov_play_internal_sample_rate(ov_play* play) {
    if (!play->device_ready) return 0;
    return play->device.playback.internalSampleRate;
}
