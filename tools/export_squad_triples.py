# SQuAD -> SHINE triple JSONL (research scaffolding; the trained artifact
# and the trainer are Zig-native). Emits {"evidence", "instruction",
# "response"} per line for `zig build shine-train`, splitting by CONTEXT
# (held-out contexts are unseen documents, the tier-2 falsifier condition).
#
#   python tools/export_squad_triples.py --parquet refs/SHINE/data/squad/*.parquet \
#       --train-out squad_train.jsonl --eval-out squad_eval.jsonl \
#       [--max-triples 4000] [--eval-contexts 25]
import argparse
import glob
import hashlib
import json

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--parquet", nargs="+", required=True)
parser.add_argument("--train-out", required=True)
parser.add_argument("--eval-out", required=True)
parser.add_argument("--max-triples", type=int, default=4000)
parser.add_argument("--eval-contexts", type=int, default=25)
args = parser.parse_args()

frames = [pd.read_parquet(p) for pattern in args.parquet for p in sorted(glob.glob(pattern))]
df = pd.concat(frames, ignore_index=True)

def answer_text(row):
    ans = row["answers"]
    if isinstance(ans, dict):
        texts = ans.get("text")
    else:
        texts = getattr(ans, "get", lambda *_: None)("text")
    if texts is None and hasattr(ans, "__getitem__"):
        try:
            texts = ans["text"]
        except Exception:
            texts = None
    if texts is None or len(texts) == 0:
        return None
    return str(texts[0])

def bucket(context):
    return int(hashlib.sha1(context.encode()).hexdigest()[:8], 16)

eval_contexts = {}
train_rows, eval_rows = [], []
for _, row in df.iterrows():
    answer = answer_text(row)
    if not answer:
        continue
    context = str(row["context"]).strip()
    question = str(row["question"]).strip()
    if not context or not question:
        continue
    triple = {"evidence": context, "instruction": question, "response": answer}
    if bucket(context) % 20 == 0 and (context in eval_contexts or len(eval_contexts) < args.eval_contexts):
        eval_contexts[context] = True
        eval_rows.append(triple)
    else:
        if len(train_rows) < args.max_triples:
            train_rows.append(triple)
    if len(train_rows) >= args.max_triples and len(eval_contexts) >= args.eval_contexts:
        break

with open(args.train_out, "w") as f:
    for t in train_rows:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
with open(args.eval_out, "w") as f:
    for t in eval_rows:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
print(f"train: {len(train_rows)} triples; eval: {len(eval_rows)} triples over {len(eval_contexts)} unseen contexts")
