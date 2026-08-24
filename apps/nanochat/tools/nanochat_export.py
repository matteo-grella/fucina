#!/usr/bin/env python3
"""Export the reference nanochat tokenizer, data, and parity oracles.

This CLI reads the reference karpathy/nanochat artifacts (the pickled tiktoken
Encoding, the token_bytes.pt tensor, and the ClimbMix parquet shards) and emits
neutral little-endian binary files that the Zig port consumes directly. The
exact byte layouts are defined in the port's FORMATS.md; this tool is the
authoritative producer of those bytes.

Emitted formats (all little-endian, magics are 8 ASCII bytes):
  tokenizer.bin    "NCTOKz01"  vocab merges + special tokens
  token_bytes.bin  "NCTKB_01"  per-token byte length (for bits-per-byte)
  ids dump         "NCIDS_01"  encode_ordinary parity fixtures
  framed docs      "NCDOC_01"  base pretraining documents, in dataloader order
  train text       "NCTXT_01"  small trainer-gate corpus (tok_train text stream)

Run with the reference nanochat venv, with NANOCHAT_BASE_DIR pointing at the
nanochat cache, e.g.:

  NANOCHAT_BASE_DIR=$HOME/.cache/nanochat \
    /path/to/nanochat/.venv/bin/python nanochat_export.py export-tokenizer --out DIR
"""

import argparse
import json
import os
import struct
import sys

# -----------------------------------------------------------------------------
# Low-level little-endian writers


def w_u32(f, v):
    f.write(struct.pack("<I", v))


def w_u16(f, v):
    f.write(struct.pack("<H", v))


def w_bytes(f, b):
    f.write(b)


# -----------------------------------------------------------------------------
# Reference nanochat access

# The 9 special tokens, in canonical order (ids 256+n_merges+0..8). This mirrors
# nanochat.tokenizer.SPECIAL_TOKENS and is cross-checked against the loaded
# Encoding's _special_tokens below.
SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]


def get_base_dir():
    base = os.environ.get("NANOCHAT_BASE_DIR")
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".cache", "nanochat")
    return base


def load_encoding():
    """Load the reference tiktoken Encoding from tokenizer.pkl."""
    import pickle

    path = os.path.join(get_base_dir(), "tokenizer", "tokenizer.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_token_bytes():
    """Load the reference token_bytes.pt (int32 tensor, length n_vocab)."""
    import torch

    path = os.path.join(get_base_dir(), "tokenizer", "token_bytes.pt")
    with open(path, "rb") as f:
        tb = torch.load(f, map_location="cpu")
    return [int(x) for x in tb.tolist()]


# -----------------------------------------------------------------------------
# Merge recovery (minbpe-style)


def _bpe(mergeable_ranks, token, max_rank):
    """Merge-rank BPE on `token` bytes using only merges with rank < max_rank.

    Repeatedly merges the adjacent pair with the lowest rank among ranks below
    max_rank until no such pair remains. For a token whose own rank is max_rank,
    this leaves exactly the two constituent tokens (left, right) that were merged
    to create it.
    """
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i in range(len(parts) - 1):
            rank = mergeable_ranks.get(parts[i] + parts[i + 1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or min_rank >= max_rank:
            break
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts


def recover_merges(mergeable_ranks):
    """Reconstruct (left_id, right_id) for every merged token, in id order.

    Returns a list indexed by merge number i (token id = 256 + i) of
    (left_id, right_id) tuples. Also validates that the concatenation of the
    recovered pair's bytes equals the merged token's bytes.
    """
    by_rank = {rank: tok for tok, rank in mergeable_ranks.items()}
    n_tokens = len(mergeable_ranks)
    # Sanity: ids 0..255 must be the single-byte tokens.
    for i in range(256):
        assert len(by_rank[i]) == 1, f"token {i} is not a single byte"
    merges = []
    for m in range(256, n_tokens):
        tok = by_rank[m]
        pair = _bpe(mergeable_ranks, tok, max_rank=m)
        assert len(pair) == 2, f"token {m} did not reduce to 2 parts: {pair}"
        left_id = mergeable_ranks[pair[0]]
        right_id = mergeable_ranks[pair[1]]
        assert by_rank[left_id] + by_rank[right_id] == tok, f"merge mismatch at {m}"
        assert left_id < m and right_id < m, f"merge {m} references future ids"
        merges.append((left_id, right_id))
    return merges


# -----------------------------------------------------------------------------
# tokenizer.bin writer (shared by export-tokenizer and train-ref)


def write_tokenizer_bin(path, merges):
    """Write NCTOKz01: merges in rank order + the canonical special tokens."""
    n_merges = len(merges)
    n_vocab = 256 + n_merges + len(SPECIAL_TOKENS)
    with open(path, "wb") as f:
        w_bytes(f, b"NCTOKz01")
        w_u32(f, 1)  # version
        w_u32(f, n_vocab)
        w_u32(f, n_merges)
        w_u32(f, len(SPECIAL_TOKENS))
        for left_id, right_id in merges:
            w_u32(f, left_id)
            w_u32(f, right_id)
        for k, name in enumerate(SPECIAL_TOKENS):
            sid = 256 + n_merges + k
            sb = name.encode("utf-8")
            w_u32(f, sid)
            w_u16(f, len(sb))
            w_bytes(f, sb)
    return n_vocab


# -----------------------------------------------------------------------------
# export-tokenizer


def cmd_export_tokenizer(args):
    enc = load_encoding()
    mergeable_ranks = enc._mergeable_ranks  # dict[bytes, int]
    special = enc._special_tokens  # dict[str, int]

    n_vocab = enc.n_vocab
    n_special = len(special)
    n_nonspecial = len(mergeable_ranks)
    n_merges = n_nonspecial - 256
    assert n_vocab == n_nonspecial + n_special, (
        f"n_vocab {n_vocab} != {n_nonspecial} + {n_special}"
    )

    # Cross-check the special tokens against the canonical order.
    for k, name in enumerate(SPECIAL_TOKENS):
        expected_id = 256 + n_merges + k
        assert name in special, f"special {name} missing from Encoding"
        assert special[name] == expected_id, (
            f"special {name} id {special[name]} != expected {expected_id}"
        )
    assert set(special.keys()) == set(SPECIAL_TOKENS), "special token set mismatch"

    merges = recover_merges(mergeable_ranks)
    assert len(merges) == n_merges, f"recovered {len(merges)} merges != {n_merges}"

    os.makedirs(args.out, exist_ok=True)

    # tokenizer.bin
    tok_path = os.path.join(args.out, "tokenizer.bin")
    assert write_tokenizer_bin(tok_path, merges) == n_vocab

    # token_bytes.bin
    token_bytes = load_token_bytes()
    assert len(token_bytes) == n_vocab, (
        f"token_bytes length {len(token_bytes)} != n_vocab {n_vocab}"
    )
    tb_path = os.path.join(args.out, "token_bytes.bin")
    with open(tb_path, "wb") as f:
        w_bytes(f, b"NCTKB_01")
        w_u32(f, n_vocab)
        for nb in token_bytes:
            w_u32(f, nb)

    # merges.txt (human-readable debugging aid)
    merges_path = os.path.join(args.out, "merges.txt")
    with open(merges_path, "w") as f:
        for i, (left_id, right_id) in enumerate(merges):
            f.write(f"{left_id} {right_id} -> {256 + i}\n")

    print(f"wrote {tok_path}")
    print(f"wrote {tb_path}")
    print(f"wrote {merges_path}")
    print(f"n_vocab={n_vocab} n_merges={n_merges} n_special={n_special}")
    print(f"first 5 merges: {merges[:5]}")
    print(f"last 5 merges: {merges[-5:]}")


# -----------------------------------------------------------------------------
# dump-ids

ADVERSARIAL = [
    "   three leading spaces",
    "word2vec is 100% better in 2026, 3.14159 and 42000 units",
    "I'm you're it's don't we'll they've",
    "你好世界 🌍 café naïve",
    "a\n\nb\tc  \n  d",
    "!!!??? ...---",
    "HTTP/2.0 (RFC-7540)!",
]


def _val_documents(n_docs):
    """First n_docs rows of the first row group of the val shard, cropped to 4000 chars."""
    import pyarrow.parquet as pq
    from nanochat.dataset import list_parquet_files

    val_path = list_parquet_files()[-1]  # last shard is val
    pf = pq.ParquetFile(val_path)
    rg = pf.read_row_group(0)
    texts = rg.column("text").to_pylist()
    return [t[:4000] for t in texts[:n_docs]]


def cmd_dump_ids(args):
    enc = load_encoding()
    items = _val_documents(args.n_docs) + list(ADVERSARIAL)

    with open(args.out, "wb") as f:
        w_bytes(f, b"NCIDS_01")
        w_u32(f, len(items))
        for text in items:
            tb = text.encode("utf-8")
            ids = enc.encode_ordinary(text)
            w_u32(f, len(tb))
            w_bytes(f, tb)
            w_u32(f, len(ids))
            for tid in ids:
                w_u32(f, tid)

    print(f"wrote {args.out}")
    print(f"n_items={len(items)} (val_docs={args.n_docs} adversarial={len(ADVERSARIAL)})")


# -----------------------------------------------------------------------------
# export-docs (framed base pretraining data)


def cmd_export_docs(args):
    from nanochat.dataset import parquets_iter_batched

    max_docs = args.max_docs
    max_chars = args.max_chars

    docs_per_rowgroup = []
    n_docs = 0
    total_chars = 0

    with open(args.out, "wb") as f:
        w_bytes(f, b"NCDOC_01")
        n_docs_offset = f.tell()
        w_u32(f, 0)  # placeholder, patched at the end

        stop = False
        for batch in parquets_iter_batched(split=args.split):
            rg_count = 0
            for text in batch:
                if max_docs is not None and n_docs >= max_docs:
                    stop = True
                    break
                if max_chars is not None and total_chars >= max_chars:
                    stop = True
                    break
                tb = text.encode("utf-8")
                w_u32(f, len(tb))
                w_bytes(f, tb)
                n_docs += 1
                rg_count += 1
                total_chars += len(text)
            docs_per_rowgroup.append(rg_count)
            if stop:
                break

        f.seek(n_docs_offset)
        w_u32(f, n_docs)

    idx_path = os.path.splitext(args.out)[0] + ".idx.json"
    with open(idx_path, "w") as f:
        json.dump({"docs_per_rowgroup": docs_per_rowgroup, "split": args.split}, f)

    print(f"wrote {args.out}")
    print(f"wrote {idx_path}")
    print(f"n_docs={n_docs} total_chars={total_chars} row_groups={len(docs_per_rowgroup)}")


# -----------------------------------------------------------------------------
# export-train-text (tok_train text stream)


def cmd_export_train_text(args):
    from nanochat.dataset import parquets_iter_batched

    doc_cap = args.doc_cap
    max_chars = args.max_chars

    n_docs = 0
    total_chars = 0

    with open(args.out, "wb") as f:
        w_bytes(f, b"NCTXT_01")
        n_docs_offset = f.tell()
        w_u32(f, 0)  # placeholder, patched at the end

        done = False
        # Reproduce scripts.tok_train.text_iterator exactly: flatten batches,
        # crop each doc to doc_cap, count cropped chars, yield, and stop once
        # nchars exceeds max_chars (checked AFTER yielding the crossing doc).
        for batch in parquets_iter_batched(split="train"):
            for doc in batch:
                doc_text = doc
                if len(doc_text) > doc_cap:
                    doc_text = doc_text[:doc_cap]
                total_chars += len(doc_text)
                tb = doc_text.encode("utf-8")
                w_u32(f, len(tb))
                w_bytes(f, tb)
                n_docs += 1
                if total_chars > max_chars:
                    done = True
                    break
            if done:
                break

        f.seek(n_docs_offset)
        w_u32(f, n_docs)

    print(f"wrote {args.out}")
    print(f"n_docs={n_docs} total_chars={total_chars}")


# -----------------------------------------------------------------------------
# train-ref (rustbpe reference training on an NCTXT_01 corpus)


def read_train_text(path):
    """Read an NCTXT_01 corpus back into a list of document strings."""
    with open(path, "rb") as f:
        data = f.read()
    assert data[:8] == b"NCTXT_01", f"bad magic in {path}"
    (n_docs,) = struct.unpack_from("<I", data, 8)
    off = 12
    docs = []
    for _ in range(n_docs):
        (doc_len,) = struct.unpack_from("<I", data, off)
        off += 4
        docs.append(data[off : off + doc_len].decode("utf-8"))
        off += doc_len
    assert off == len(data), f"trailing bytes in {path}"
    return docs


def cmd_train_ref(args):
    import rustbpe
    from nanochat.tokenizer import SPLIT_PATTERN, SPECIAL_TOKENS as NC_SPECIAL

    assert NC_SPECIAL == SPECIAL_TOKENS, "special token list drifted from nanochat"

    docs = read_train_text(args.input)

    # Mirror RustBPETokenizer.train_from_iterator: the specials are appended
    # after training, so vocab_size - len(SPECIAL_TOKENS) tokens are trained.
    vocab_size_no_special = args.vocab - len(SPECIAL_TOKENS)
    assert vocab_size_no_special >= 256, f"vocab {args.vocab} too small"
    tokenizer = rustbpe.Tokenizer()
    tokenizer.train_from_iterator(iter(docs), vocab_size_no_special, pattern=SPLIT_PATTERN)

    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    merges = recover_merges(mergeable_ranks)
    assert len(merges) == vocab_size_no_special - 256, (
        f"recovered {len(merges)} merges != {vocab_size_no_special - 256}"
    )

    n_vocab = write_tokenizer_bin(args.out, merges)
    print(f"wrote {args.out}")
    print(f"n_docs={len(docs)} n_vocab={n_vocab} n_merges={len(merges)}")
    print(f"first 5 merges: {merges[:5]}")
    print(f"last 5 merges: {merges[-5:]}")


# -----------------------------------------------------------------------------
# SFT mixture (SmolTalk / MMLU / GSM8K) — shared by the SFT exporters below.
#
# Mirrors scripts/chat_sft.py's train/val TaskMixture construction exactly
# (train = SmolTalk(train) + MMLU(aux)x3 + GSM8K(train)x4;
#  val   = SmolTalk(test) + MMLU(test)[:5200] + GSM8K(test)[:420]).
# TaskMixture applies random.Random(42).shuffle over its index_map, so the
# mixture-ordered conversation stream is deterministic; the Zig port consumes
# that materialized order (ALL shuffle order is baked in Python).


def build_mixture(split):
    from tasks.common import TaskMixture
    from tasks.smoltalk import SmolTalk
    from tasks.mmlu import MMLU
    from tasks.gsm8k import GSM8K

    if split == "train":
        tasks = [
            SmolTalk(split="train"),
            *[MMLU(subset="all", split="auxiliary_train") for _ in range(3)],
            *[GSM8K(subset="main", split="train") for _ in range(4)],
        ]
    else:
        tasks = [
            SmolTalk(split="test"),
            MMLU(subset="all", split="test", stop=5200),
            GSM8K(subset="main", split="test", stop=420),
        ]
    return TaskMixture(tasks)


def cmd_export_sft_mixture(args):
    mixture = build_mixture(args.split)
    n = len(mixture)
    cap = n if args.max_convs is None else min(n, args.max_convs)
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(cap):
            conv = mixture[i]
            # Emit ONLY the messages (content stays a string, or a list of
            # {"type","text"} parts for GSM8K tool use). The extra MMLU keys
            # (subject/letters) are irrelevant to render_conversation.
            rec = {"messages": conv["messages"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    print(f"mixture_size={n} emitted={cap} split={args.split}")


def cmd_dump_render(args):
    from nanochat.tokenizer import get_tokenizer

    tok = get_tokenizer()
    mixture = build_mixture(args.split)
    N = min(len(mixture), args.n)
    with open(args.out, "wb") as f:
        w_u32(f, N)
        for i in range(N):
            conv = mixture[i]
            ids, mask = tok.render_conversation(conv)  # default max_tokens=2048
            assert len(ids) == len(mask)
            w_u32(f, len(ids))
            for tid in ids:
                w_u32(f, tid)
            f.write(bytes(bytearray(mask)))  # one u8 per token (0/1)
    print(f"wrote {args.out}")
    print(f"n_convs={N} split={args.split}")


# -----------------------------------------------------------------------------
# dump-base-batches — run the reference base BOS-bestfit loader, dump batches.


def cmd_dump_base_batches(args):
    import torch  # noqa: F401  (loader builds CPU torch tensors)
    from nanochat.tokenizer import get_tokenizer
    from nanochat.dataloader import (
        tokenizing_distributed_data_loader_with_state_bos_bestfit as loader_fn,
    )

    tok = get_tokenizer()
    gen = loader_fn(tok, args.B, args.T, args.split, device="cpu")
    K = args.n_batches
    with open(args.out, "wb") as f:
        w_u32(f, K)
        w_u32(f, args.B)
        w_u32(f, args.T)
        for _ in range(K):
            inputs, targets, state = next(gen)
            f.write(inputs.cpu().numpy().astype("<i4").tobytes())
            f.write(targets.cpu().numpy().astype("<i4").tobytes())
            f.write(struct.pack("<q", int(state["pq_idx"])))
            f.write(struct.pack("<q", int(state["rg_idx"])))
            f.write(struct.pack("<q", int(state["epoch"])))
    print(f"wrote {args.out}")
    print(f"K={K} B={args.B} T={args.T} split={args.split}")


# -----------------------------------------------------------------------------
# dump-sft-batches — run the reference SFT bestfit-pad generator, dump batches.
#
# scripts/chat_sft.py defines sft_data_generator_bos_bestfit as a closure over
# the training-script globals (args/tokenizer/datasets/device), so it cannot be
# imported without running that script. This is a faithful transcription of it
# for the single-device (ddp_rank=0, ddp_world_size=1) val path; the emitted
# (inputs, targets) tensors are byte-identical to the reference.


def _sft_generator(tok, dataset, device_batch_size, max_seq_len, buffer_size=100):
    import torch

    ddp_rank, ddp_world_size = 0, 1
    dataset_size = len(dataset)
    assert dataset_size > 0
    row_capacity = max_seq_len + 1
    bos_token = tok.get_bos_token_id()

    conv_buffer = []
    cursor = ddp_rank
    epoch = 1

    def refill_buffer():
        nonlocal cursor, epoch
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor]
            ids, mask = tok.render_conversation(conversation)
            conv_buffer.append((ids, mask))
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1

    while True:
        rows = []
        mask_rows = []
        row_lengths = []
        for _ in range(device_batch_size):
            row = []
            mask_row = []
            padded = False
            while len(row) < row_capacity:
                while len(conv_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - len(row)
                best_idx = -1
                best_len = 0
                for i, (conv, _) in enumerate(conv_buffer):
                    conv_len = len(conv)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len
                if best_idx >= 0:
                    conv, conv_mask = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    mask_row.extend(conv_mask)
                else:
                    content_len = len(row)
                    row.extend([bos_token] * remaining)
                    mask_row.extend([0] * remaining)
                    padded = True
                    break
            if padded:
                row_lengths.append(content_len)
            else:
                row_lengths.append(row_capacity)
            rows.append(row[:row_capacity])
            mask_rows.append(mask_row[:row_capacity])

        batch_tensor = torch.tensor(rows, dtype=torch.long)
        inputs = batch_tensor[:, :-1].to(dtype=torch.int32).contiguous()
        targets = batch_tensor[:, 1:].to(dtype=torch.int64).contiguous()
        mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)
        mask_targets = mask_tensor[:, 1:]
        targets[mask_targets == 0] = -1
        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[i, content_len - 1:] = -1
        yield inputs, targets


def cmd_dump_sft_batches(args):
    from nanochat.tokenizer import get_tokenizer

    tok = get_tokenizer()
    dataset = build_mixture("val")
    gen = _sft_generator(tok, dataset, args.B, args.T)
    K = args.n_batches
    with open(args.out, "wb") as f:
        w_u32(f, K)
        w_u32(f, args.B)
        w_u32(f, args.T)
        for _ in range(K):
            inputs, targets = next(gen)
            f.write(inputs.cpu().numpy().astype("<i4").tobytes())
            f.write(targets.cpu().numpy().astype("<i8").tobytes())
    print(f"wrote {args.out}")
    print(f"K={K} B={args.B} T={args.T} mixture_size={len(dataset)}")


# -----------------------------------------------------------------------------
# main / argparse dispatch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("export-tokenizer", help="write tokenizer.bin + token_bytes.bin + merges.txt")
    p.add_argument("--out", required=True, help="output directory")
    p.set_defaults(func=cmd_export_tokenizer)

    p = sub.add_parser("dump-ids", help="write NCIDS_01 encode-parity fixture")
    p.add_argument("--out", required=True, help="output file")
    p.add_argument("--n-docs", type=int, default=200, help="val documents to sample")
    p.set_defaults(func=cmd_dump_ids)

    p = sub.add_parser("export-docs", help="write NCDOC_01 framed base docs + idx.json")
    p.add_argument("--split", required=True, choices=["train", "val"])
    p.add_argument("--out", required=True, help="output file")
    p.add_argument("--max-docs", type=int, default=20000, help="cap on number of docs")
    p.add_argument("--max-chars", type=int, default=None, help="cap on total characters")
    p.set_defaults(func=cmd_export_docs)

    p = sub.add_parser("train-ref", help="train rustbpe on an NCTXT_01 corpus → NCTOKz01 reference")
    p.add_argument("--input", required=True, help="NCTXT_01 corpus file")
    p.add_argument("--vocab", type=int, required=True, help="vocab size INCLUDING the 9 specials")
    p.add_argument("--out", required=True, help="output tokenizer.bin path")
    p.set_defaults(func=cmd_train_ref)

    p = sub.add_parser("export-train-text", help="write NCTXT_01 trainer-gate corpus")
    p.add_argument("--out", required=True, help="output file")
    p.add_argument("--max-chars", type=int, default=5_000_000, help="max cropped chars")
    p.add_argument("--doc-cap", type=int, default=10_000, help="max chars per document")
    p.set_defaults(func=cmd_export_train_text)

    p = sub.add_parser("export-sft-mixture", help="write JSONL of mixture-ordered SFT conversations")
    p.add_argument("--split", required=True, choices=["train", "val"])
    p.add_argument("--out", required=True, help="output .jsonl file")
    p.add_argument("--max-convs", type=int, default=None, help="cap number of conversations")
    p.set_defaults(func=cmd_export_sft_mixture)

    p = sub.add_parser("dump-render", help="dump render_conversation {ids,mask} for the first N mixture convs")
    p.add_argument("--split", required=True, choices=["train", "val"])
    p.add_argument("--out", required=True, help="output file")
    p.add_argument("--n", type=int, default=64, help="number of conversations")
    p.set_defaults(func=cmd_dump_render)

    p = sub.add_parser("dump-base-batches", help="dump first K batches of the base BOS-bestfit loader")
    p.add_argument("--B", type=int, required=True, help="batch size (rows)")
    p.add_argument("--T", type=int, required=True, help="sequence length")
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--out", required=True, help="output file")
    p.add_argument("--n-batches", type=int, default=8, help="number of batches K")
    p.set_defaults(func=cmd_dump_base_batches)

    p = sub.add_parser("dump-sft-batches", help="dump first K batches of the SFT bestfit-pad loader (val mixture)")
    p.add_argument("--B", type=int, required=True, help="batch size (rows)")
    p.add_argument("--T", type=int, required=True, help="sequence length")
    p.add_argument("--out", required=True, help="output file")
    p.add_argument("--n-batches", type=int, default=4, help="number of batches K")
    p.set_defaults(func=cmd_dump_sft_batches)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
