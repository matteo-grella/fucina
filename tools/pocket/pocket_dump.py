"""Reference-activation dump for the Pocket TTS parity gate.

Monkeypatches the stock pocket-tts checkout to capture every tensor
src/llm/pockettts/pocket_tests.zig compares against: text tokens, backbone
transformer in/out, the flow-net noise and sampled latent, the Mimi decoder
stages (latent in, upsampled, transformer out, PCM chunk), and the final
audio. The torch seed is fixed so the run is reproducible.

Out-of-tree by design: the reference stays stock (tools/fetch_refs.sh pins
it), and this harness runs against it rather than patching it.

    tools/fetch_refs.sh pocket-tts
    python3 -m venv refs/pocket-tts-venv
    refs/pocket-tts-venv/bin/pip install -e refs/pocket-tts
    PYTHONPATH=refs/pocket-tts refs/pocket-tts-venv/bin/python \
        tools/pocket/pocket_dump.py refs/pocket-tts-dumps

The output directory defaults to refs/pocket-tts-dumps, which is where
pocket_tests.zig looks; the test skips cleanly when it is absent.
"""

import os
import sys

import numpy as np
import torch

torch.manual_seed(42)

from pocket_tts.models.tts_model import TTSModel, prepare_text_prompt
from pocket_tts.models import flow_lm as flow_lm_mod
from pocket_tts.modules.mimi_transformer import StreamingTransformer

DUMP_DIR = sys.argv[1] if len(sys.argv) > 1 else "refs/pocket-tts-dumps"
TEXT = "Parity check."
VOICE = "alba"

os.makedirs(DUMP_DIR, exist_ok=True)
dumps = {}


def save(name, t):
    dumps.setdefault(name, []).append(t.detach().to(torch.float32).cpu().numpy().copy())


# backbone transformer, per forward call
orig_tf_forward = StreamingTransformer.forward


def tf_forward(self, x, model_state, *a, **k):
    save("backbone_in", x)
    out = orig_tf_forward(self, x, model_state, *a, **k)
    save("backbone_out", out)
    return out


StreamingTransformer.forward = tf_forward

# flow-net integration: the noise it starts from and the latent it samples
orig_lsd = flow_lm_mod.lsd_decode


def lsd_decode(v_t, x_0, num_steps=1):
    save("flow_noise_x0", x_0)
    out = orig_lsd(v_t, x_0, num_steps)
    save("flow_latent_x1", out)
    return out


flow_lm_mod.lsd_decode = lsd_decode

model = TTSModel.load_model()

# Mimi decoder stages (instance-level: these are bound methods on the loaded
# model, so they must be patched after load_model()).
mimi = model.mimi
orig_upsample = mimi.upsample.forward
orig_dec_tf = mimi.decoder_transformer.forward
orig_dec = mimi.decoder.forward


def upsample_fwd(x, model_state=None):
    save("mimi_latent_in", x)
    out = orig_upsample(x, model_state)
    save("mimi_upsampled", out)
    return out


def dec_tf_fwd(x, model_state=None):
    outs = orig_dec_tf(x, model_state)
    save("mimi_dec_transformer_out", outs[0])
    return outs


def dec_fwd(x, model_state=None):
    out = orig_dec(x, model_state)
    save("mimi_pcm_chunk", out)
    return out


mimi.upsample.forward = upsample_fwd
mimi.decoder_transformer.forward = dec_tf_fwd
mimi.decoder.forward = dec_fwd

voice_state = model.get_state_for_audio_prompt(VOICE)
audio = model.generate_audio(voice_state, TEXT)

# Re-tokenize for the record: generate_audio does this internally with the
# same flags, so the dumped ids are what the backbone actually consumed.
cleaned, _ = prepare_text_prompt(
    TEXT, pad_with_spaces_for_short_inputs=True, remove_semicolons=True
)
tokens = model.flow_lm.conditioner.tokenizer(cleaned).tokens
np.save(os.path.join(DUMP_DIR, "text_tokens.npy"), tokens.numpy())

# The gate reads the first few calls of each stage; capping keeps the dump
# directory small without weakening it (later frames exercise the same code).
for name, arrs in dumps.items():
    for i, a in enumerate(arrs[:4]):
        np.save(os.path.join(DUMP_DIR, f"{name}_{i:03d}.npy"), a)
    print(f"{name}: {len(arrs)} calls, first shape {arrs[0].shape}")

np.save(os.path.join(DUMP_DIR, "final_audio.npy"), audio.numpy())
print("audio samples:", audio.shape, "sr:", model.sample_rate)
