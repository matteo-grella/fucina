#!/bin/sh
# Fetch every weight examples/voiceagent needs, into the layout its README
# uses. The agent is a four-stage cascade, so this is the one example where
# collecting the models by hand is genuinely fiddly: five artifacts from four
# sources, one of which needs a local conversion.
#
#   tools/fetch_voice_models.sh            # fast tier (Pocket TTS, no codec)
#   tools/fetch_voice_models.sh quality    # Qwen3-TTS CustomVoice + codec
#   tools/fetch_voice_models.sh both
#
# Requires the Hugging Face CLI (`pip install -U huggingface_hub`; the binary
# is `hf`, or `huggingface-cli` on older installs). The Pocket tier also needs
# python with `safetensors` and `sentencepiece` for the checkpoint conversion —
# kyutai ships safetensors, not GGUF, so tools/pocket/pocket_to_gguf.py packs
# it. Every other artifact is a direct download.
#
# Weights are not redistributed by this repository and carry their own terms;
# see docs/THIRD-PARTY-NOTICES.md. Re-running is cheap: the CLI skips files it
# already has, and the conversion is skipped if its output exists.
set -eu
cd "$(dirname "$0")/.."

tier="${1:-fast}"
case "$tier" in
fast | quality | both) ;;
*)
    echo "usage: tools/fetch_voice_models.sh [fast|quality|both]" >&2
    exit 2
    ;;
esac

if command -v hf >/dev/null 2>&1; then
    HF=hf
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF=huggingface-cli
else
    echo "error: no Hugging Face CLI on PATH — pip install -U huggingface_hub" >&2
    exit 1
fi

get() { # repo file dest
    echo "  $3/$2"
    "$HF" download "$1" "$2" --local-dir "$3" >/dev/null
}

echo "[1/3] speech recognition — parakeet streaming EOU"
get mudler/parakeet-cpp-gguf realtime_eou_120m-v1-f16.gguf models/parakeet

echo "[2/3] chat — Qwen3 1.7B (the latency-dominant stage; see the README tiers)"
get unsloth/Qwen3-1.7B-GGUF Qwen3-1.7B-Q4_K_M.gguf models

echo "[3/3] echo canceller — LocalVQE compact GTCRN-AEC"
get LocalAI-io/LocalVQE localvqe-pi-aec-v1-49k-f32.gguf models/aec
# The agent's default --aec path is models/aec/gtcrn_aec.gguf. Link rather
# than rename, so the upstream filename stays visible.
[ -e models/aec/gtcrn_aec.gguf ] || ln -s localvqe-pi-aec-v1-49k-f32.gguf models/aec/gtcrn_aec.gguf

if [ "$tier" = fast ] || [ "$tier" = both ]; then
    echo "[tts] Pocket TTS — checkpoint + conversion"
    "$HF" download kyutai/pocket-tts-without-voice-cloning --local-dir models/pocket-tts >/dev/null
    if [ -e models/pocket-tts/pocket-tts-english-v2.gguf ]; then
        echo "  models/pocket-tts/pocket-tts-english-v2.gguf (already converted)"
    else
        echo "  converting safetensors -> GGUF ..."
        python3 tools/pocket/pocket_to_gguf.py
    fi
fi

if [ "$tier" = quality ] || [ "$tier" = both ]; then
    echo "[tts] Qwen3-TTS — talker + codec"
    get Serveurperso/Qwen3-TTS-GGUF qwen-talker-0.6b-customvoice-Q8_0.gguf models/qwen3-tts
    get Serveurperso/Qwen3-TTS-GGUF qwen-tokenizer-12hz-F32.gguf models/qwen3-tts
fi

echo
echo "done. run it with:"
echo
if [ "$tier" = quality ]; then
    cat <<'EOF'
  zig build voiceagent -Doptimize=ReleaseFast -- \
      --asr   models/parakeet/realtime_eou_120m-v1-f16.gguf \
      --chat  models/Qwen3-1.7B-Q4_K_M.gguf \
      --tts   models/qwen3-tts/qwen-talker-0.6b-customvoice-Q8_0.gguf \
      --codec models/qwen3-tts/qwen-tokenizer-12hz-F32.gguf
EOF
else
    cat <<'EOF'
  zig build voiceagent -Doptimize=ReleaseFast -- \
      --asr   models/parakeet/realtime_eou_120m-v1-f16.gguf \
      --chat  models/Qwen3-1.7B-Q4_K_M.gguf \
      --tts   models/pocket-tts/pocket-tts-english-v2.gguf \
      --voice alba
EOF
fi
