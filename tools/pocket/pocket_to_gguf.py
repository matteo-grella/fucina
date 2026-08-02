#!/usr/bin/env python3
"""Pocket TTS v2 checkpoint -> single fucina GGUF.

Packs the FlowLM + flow head + Mimi DECODER path (encoder skipped: TTS only
decodes), the SentencePiece tokenizer (pieces + scores), per-voice KV-cache
prefixes, and the config constants. All tensors f32. GGUF dims are ne-order
(reversed torch shape).
"""
import struct, sys, io
from safetensors import safe_open
import sentencepiece as spm

MODEL = "models/pocket-tts/languages/english/model.safetensors"
TOKENIZER = "models/pocket-tts/languages/english/tokenizer.model"
VOICES = {
    "alba": "models/pocket-tts/languages/english/embeddings/alba.safetensors",
    "marius": "models/pocket-tts/languages/english/embeddings/marius.safetensors",
}
OUT = "models/pocket-tts/pocket-tts-english-v2.gguf"

GGUF_STR, GGUF_U32, GGUF_F32, GGUF_ARR, GGUF_U64 = 8, 4, 6, 9, 10

def w_str(b, s):
    e = s.encode()
    b += struct.pack("<Q", len(e)); b += e
    return b

kv = []  # (key, type, value)
def add_u32(k, v): kv.append((k, GGUF_U32, v))
def add_f32(k, v): kv.append((k, GGUF_F32, v))
def add_str(k, v): kv.append((k, GGUF_STR, v))

add_str("general.architecture", "pocket-tts")
add_str("general.name", "Pocket TTS english v2 (kyutai, CC-BY-4.0)")
add_u32("pocket.d_model", 1024); add_u32("pocket.layers", 6)
add_u32("pocket.heads", 16); add_u32("pocket.ffn", 4096)
add_f32("pocket.ln_eps", 1e-5); add_f32("pocket.rope_max_period", 10000.0)
add_u32("pocket.latent_dim", 32)
add_u32("pocket.flow.hidden", 512); add_u32("pocket.flow.blocks", 6)
add_f32("pocket.flow.ln_eps", 1e-6); add_u32("pocket.flow.time_freqs", 128)
add_f32("pocket.default_temperature", 0.3)
add_u32("pocket.text.vocab", 4000)
add_u32("pocket.mimi.d_model", 512); add_u32("pocket.mimi.layers", 2)
add_u32("pocket.mimi.heads", 8); add_u32("pocket.mimi.ffn", 2048)
add_u32("pocket.mimi.context", 250)
add_u32("pocket.mimi.frame_size", 1920); add_u32("pocket.mimi.sample_rate", 24000)

sp = spm.SentencePieceProcessor(model_file=TOKENIZER)
n = sp.get_piece_size()
assert n == 4000, n
pieces = [sp.id_to_piece(i) for i in range(n)]
scores = [sp.get_score(i) for i in range(n)]
kv.append(("tokenizer.pocket.tokens", GGUF_ARR, (GGUF_STR, pieces)))
kv.append(("tokenizer.pocket.scores", GGUF_ARR, (GGUF_F32, scores)))

tensors = []  # (name, ne_dims, np.float32 array bytes)
def add_tensor(name, t):
    import numpy as np
    a = t.float().numpy() if hasattr(t, "float") else t
    a = np.ascontiguousarray(a, dtype=np.float32)
    ne = list(reversed(a.shape))
    tensors.append((name, ne, a.tobytes()))

f = safe_open(MODEL, "pt")
skip_prefixes = ("mimi.encoder.", "mimi.encoder_transformer.", "mimi.downsample.")
# LayerScale folds into the FOLLOWING linear's output rows (scale is a
# per-output-channel constant; the linear has no bias) — zero runtime ops.
scales = {}
for k in f.keys():
    if k.endswith("layer_scale_1.scale") or k.endswith("layer_scale_2.scale"):
        scales[k] = f.get_tensor(k).float()
for k in sorted(f.keys()):
    if any(k.startswith(p) for p in skip_prefixes):
        continue
    if k in scales:
        continue  # folded below
    t = f.get_tensor(k)
    s1 = k.replace("self_attn.out_proj.weight", "layer_scale_1.scale")
    s2 = k.replace("linear2.weight", "layer_scale_2.scale")
    if k.endswith("self_attn.out_proj.weight") and s1 in scales:
        t = t.float() * scales[s1][:, None]
    elif k.endswith("linear2.weight") and s2 in scales:
        t = t.float() * scales[s2][:, None]
    add_tensor(k, t)

for vname, vpath in VOICES.items():
    vf = safe_open(vpath, "pt")
    for k in sorted(vf.keys()):
        if k.endswith("/cache"):
            layer = k.split(".")[2]
            t = vf.get_tensor(k)  # [2,1,T,16,64]
            assert t.shape[0] == 2 and t.shape[1] == 1
            add_tensor(f"voice.{vname}.cache.{layer}", t[:, 0])  # [2,T,16,64]
        elif k.endswith("/offset"):
            layer = k.split(".")[2]
            off = int(vf.get_tensor(k).item())
            if layer == "0":
                add_u32(f"pocket.voice.{vname}.len", off)

buf = io.BytesIO()
b = bytearray()
b += b"GGUF"; b += struct.pack("<IQQ", 3, len(tensors), len(kv))
for key, typ, val in kv:
    b = w_str(b, key); b += struct.pack("<I", typ)
    if typ == GGUF_U32: b += struct.pack("<I", val)
    elif typ == GGUF_F32: b += struct.pack("<f", val)
    elif typ == GGUF_STR: b = w_str(b, val)
    elif typ == GGUF_ARR:
        et, items = val
        b += struct.pack("<IQ", et, len(items))
        for it in items:
            if et == GGUF_STR: b = w_str(b, it)
            elif et == GGUF_F32: b += struct.pack("<f", it)

offset = 0
infos = bytearray()
for name, ne, data in tensors:
    infos = w_str(infos, name)
    infos += struct.pack("<I", len(ne))
    for d in ne: infos += struct.pack("<Q", d)
    infos += struct.pack("<IQ", 0, offset)  # type 0 = f32
    offset += len(data)
    offset = (offset + 31) & ~31
b += infos

pad = (-(len(b)) % 32)
b += b"\x00" * pad
for name, ne, data in tensors:
    b += data
    b += b"\x00" * ((-len(data)) % 32)

open(OUT, "wb").write(bytes(b))
print(f"wrote {OUT}: {len(tensors)} tensors, {len(kv)} kv, {len(b)/1e6:.1f} MB")
