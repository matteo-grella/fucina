# nanochat port — neutral interchange formats (authoritative)

Both the Python export/instrumentation tools (`tools/nanochat_export.py`,
`tools/nanochat_dump.py`) and the Zig port MUST use these exact formats. All
integers little-endian. Keep this file in sync with both sides.

## tokenizer.bin  (magic "NCTOKz01")
Fully reconstructs the vocab from byte tokens + merges + specials.
```
[8]  magic  = "NCTOKz01"
u32  version = 1
u32  n_vocab        # total incl specials (e.g. 32768)
u32  n_merges       # e.g. 32503
u32  n_special      # e.g. 9
# merges in rank order; merge i produces token id (256 + i)
n_merges × { u32 left_id ; u32 right_id }
# specials appended after byte+merge tokens; id = 256 + n_merges + k
n_special × { u32 id ; u16 str_len ; [str_len] bytes }   # utf8 special-token string
```
Reconstruction: tokens 0..255 = single byte; token (256+i) = bytes[left]+bytes[right];
specials as given. Invariant: 256 + n_merges + n_special == n_vocab.

## token_bytes.bin  (magic "NCTKB_01") — for bits-per-byte eval
```
[8]  magic = "NCTKB_01"
u32  n_vocab
n_vocab × u32   # byte length of token id, or 0 if special (not counted)
```

## ids dump  (magic "NCIDS_01") — encode-parity fixtures
A list of (text, expected_ids). Text is raw utf8 bytes exactly as fed to encode
(encode_ordinary — no bos/specials prepended). Goldens instance: ids_parity.bin.
```
[8]  magic = "NCIDS_01"
u32  n_items
n_items × { u32 text_len ; [text_len] bytes ; u32 n_ids ; n_ids × u32 }
```

## train text stream  (magic "NCTXT_01") — small deterministic trainer-gate corpus
The concatenation of pretokenizer INPUT documents (each doc = one string passed to
the tokenizer trainer's text_iterator, already doc_cap-cropped). Used to train BOTH
Python rustbpe (`train-ref`) and the Zig trainer (`tok-train --input`) on an
identical stream for an exact merge-list gate.
```
[8]  magic = "NCTXT_01"
u32  n_docs
n_docs × { u32 doc_len ; [doc_len] bytes }
```

## framed docs  (magic "NCDOC_01") — base pretraining data export
Preserves exact (shard, row_group, doc) order so BOS-bestfit packing is reproducible.
Docs are raw utf8 text (NOT tokenized); Zig tokenizes with its own tokenizer.bin.
```
[8]  magic = "NCDOC_01"
u32  n_docs
# docs in the exact iteration order of parquets_iter_batched(split) →
#   for each parquet (sorted; train = all-but-last, val = last),
#     for each row_group in order, each 'text' column entry in order.
n_docs × { u32 text_len ; [text_len] bytes }
```
`export-docs --split {train,val}` writes any `--out` name (goldens convention:
base_train_small.bin / base_val.bin) plus a REQUIRED sidecar whose name replaces
the ".bin" extension: `<base>.idx.json` (Python `os.path.splitext(out)[0] +
".idx.json"`; `data.zig` derives the same name), containing
`{"docs_per_rowgroup": [doc counts per row group, in order], "split": ...}`.
The Zig reader validates sum(docs_per_rowgroup) == n_docs and uses the row-group
boundaries to reconstruct the reference dataloader resume state
(pq_idx/rg_idx/epoch).

## SFT mixture JSONL — one conversation per line
```
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}, ...]}
```
`export-sft-mixture --split {train,val}` emits ONE pre-mixed file per split
(sft_mixture_train.jsonl / sft_mixture_val.jsonl): the reference TaskMixture's
conversations (train = SmolTalk(train) + MMLU(aux)×3 + GSM8K(train)×4; val =
SmolTalk(test) + MMLU(test)[:5200] + GSM8K(test)[:420]) materialized in final
mixture order. ALL shuffle order is baked in Python — each task's HubDataset
shuffle (numpy default_rng(seed).permutation) and the TaskMixture interleave
(random.Random(42).shuffle over its index_map) — so Zig reads the lines in file
order (`sft --mixture/--val-mixture`) with no shuffle of its own. `role` is
system|user|assistant; `content` is a string, or a list of parts for tool use
(GSM8K): {"type":"text"|"python"|"python_output","text":"..."}. Only the
"messages" key is emitted; blank lines are skipped. The Zig SFT loader applies
the same render_conversation masking.

## oracle dumps (parity) — safetensors + json + framed binaries
From `nanochat_dump.py` (configs d6/d2; float32 CPU, dynamo disabled):
- init_d{N}.safetensors : model state_dict right after init_weights() (torch
  names) + init_d{N}.config.json (geometry, window_sizes, rotary base/len,
  rms_norm_eps incl. its discovery record).
- fixed_batch_d{N}.bin : { u32 B ; u32 T ; B*T u32 inputs ; B*T i32 targets
  (−1 = ignore_index) }.
- fwd_oracle_d6.safetensors : per-layer activations + logits + mean loss for the
  fixed batch; a --loss-reduction none run adds the per-token (B,T) loss as
  tensor "loss_none" (fwd_oracle_d6_none.safetensors).
- grad_oracle_d6.safetensors : every param's .grad (+ loss) after one backward
  on the fixed batch.
- optstep_d{N}_{k}.safetensors : params + optimizer state after k optim steps
  (k ∈ {1,10}; adamw exp_avg/exp_avg_sq per param, muon momentum_buffer/
  second_momentum_buffer keyed by the group's first param name), plus
  optstep_d{N}_schedule.json (hyperparameters, group metadata, per-step
  lr/momentum/weight-decay rows).
- trace_batches_d6.bin : { u32 n_steps ; u32 B ; u32 T ; per step B*T u32
  inputs + B*T i32 targets } ; loss_trace_d6.json = list of per-step pre-step
  mean losses (+ loss_trace_d6.meta.json provenance).
- bpb_oracle.json : reference evaluate_bpb over the trace batches
  ({bpb, total_nats, total_bytes, ...}).
- SFT trace, from the trained base checkpoint (base_ckpt_d6_step2500.safetensors)
  on the val mixture: trace_batches_sft_d6.bin { u32 n_steps ; u32 B ; u32 T ;
  per step B*T i32 inputs + B*T i32 targets (−1 = masked) },
  loss_trace_sft_d6.json (per-step pre-step masked mean loss), and
  sft_schedule_d6.json (SFT hyperparameters incl. init_lr_frac + schedule rows).
- greedy_stream_d6.json / greedy_trained_d6_step2500.json :
  {prompt, prompt_ids, out_ids} temp=0 decode (init / trained checkpoint).

Loader fixtures from `nanochat_export.py` (consumed by the data parity gates):
- render_val.bin : u32 N ; N × { u32 n_ids ; n_ids × u32 ids ; n_ids × u8 mask }
  — render_conversation ids + supervision mask (0/1) for the first N mixture
  conversations.
- base_batches.bin : u32 K ; u32 B ; u32 T ; K × { B*T i32 inputs ; B*T i32
  targets ; i64 pq_idx ; i64 rg_idx ; i64 epoch } — first K batches (+ resume
  state) of the reference base BOS-bestfit loader.
- sft_batches.bin : u32 K ; u32 B ; u32 T ; K × { B*T i32 inputs ; B*T i64
  targets (−1 = masked) } — first K batches of the reference SFT bestfit-pad
  generator.
