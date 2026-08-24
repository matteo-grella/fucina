/* Single source of truth for the MA_NO_* config seen by every TU that
 * includes the vendored third_party/miniaudio.h (the MINIAUDIO_IMPLEMENTATION
 * build in audio_shim.c and omnivoice's declaration-mode play_shim.c). On the
 * pinned miniaudio 0.11.25 this macro set does not alter the shared struct
 * layouts; sharing the block guards the cross-TU ABI against a future header
 * version bump that reintroduces config-conditional members. */
#ifndef FUCINA_MINIAUDIO_CONFIG_H
#define FUCINA_MINIAUDIO_CONFIG_H

#define MA_NO_DECODING
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_ENGINE
#define MA_NO_NODE_GRAPH
#ifdef __APPLE__
/* Link CoreAudio/AudioToolbox directly (no dlopen); see build.zig. */
#define MA_NO_RUNTIME_LINKING
#endif

#endif /* FUCINA_MINIAUDIO_CONFIG_H */
