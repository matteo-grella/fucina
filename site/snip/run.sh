# grab a small model and talk to it
hf download Qwen/Qwen3-0.6B-GGUF \
    Qwen3-0.6B-Q8_0.gguf --local-dir models
zig build qwen3 -Doptimize=ReleaseFast -- \
    models/Qwen3-0.6B-Q8_0.gguf \
    --chat "What is the capital of France?" \
    --no-think

# or serve it to any OpenAI client (SSE, JSON)
zig build lmserve -Doptimize=ReleaseFast -- \
    models/Qwen3-0.6B-Q8_0.gguf --port 8080
