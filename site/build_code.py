#!/usr/bin/env python3
"""Extract REAL Zig code from this repo (or a snip/ file) and emit a
syntax-highlighted token stream for the site (writes <out.js> defining
window.__CODE_<VAR>). The on-screen code is therefore verifiably the repo's
own — never mocked up.

Usage (from site/):
  python3 build_code.py <src_file> <start_line> <end_line> <out.js> [VARNAME]
  # src_file: absolute, or relative to the repo root. Lines are 1-based inclusive.
Examples:
  python3 build_code.py "$PWD/snip/hero.zig" 1 11 assets/code_hero.js HERO
  python3 build_code.py src/ag/tensor.zig 40 72 assets/code_dot.js DOT
"""
import re, json, sys, os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src, a, b, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
var = sys.argv[5] if len(sys.argv) > 5 else "MAIN"
path = src if os.path.isabs(src) else os.path.join(REPO, src)
lines = open(path).read().split('\n')[a - 1:b]
# if the slice is a fenced block, drop the fences
lines = [l for l in lines if not l.strip().startswith('```')]

KW = set('const var fn try defer return struct pub if else while for comptime error and or inline align switch break continue orelse catch unreachable test enum union'.split())
def tok_line(line):
    cpos = line.find('//')
    code_part = line if cpos < 0 else line[:cpos]
    com = None if cpos < 0 else line[cpos:]
    raw = re.findall(r'"[^"]*"|\.\{|\.[A-Za-z_]\w*|[A-Za-z_]\w*|@[A-Za-z_]\w*|\d+\.?\d*|\s+|[^\sA-Za-z0-9_]', code_part)
    toks = []
    for j, s in enumerate(raw):
        if not s.strip(): c = 'txt'
        elif s.startswith('"'): c = 'str'
        elif s == '.{': c = 'punct'
        elif s.startswith('@'): c = 'fn'
        elif s.startswith('.') and len(s) > 1 and (s[1].isalpha() or s[1] == '_'):
            nxt = next((raw[k] for k in range(j + 1, len(raw)) if raw[k].strip()), '')
            c = 'fn' if nxt == '(' else 'tag'
        elif s in KW: c = 'key'
        elif s[:1].isupper(): c = 'type'
        elif re.match(r'\d', s): c = 'num'
        elif re.match(r'[A-Za-z_]', s): c = 'txt'
        else: c = 'punct'
        toks.append([s, c])
    if com is not None: toks.append([com, 'com'])
    return toks

model = [tok_line(l) for l in lines]
os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
with open(out, 'w') as f:
    f.write(f'window.__CODE_{var}=' + json.dumps(model, ensure_ascii=False) + ';\n')
print(f'wrote {out}  ({len(model)} lines from {src}:{a}-{b} as __CODE_{var})')
