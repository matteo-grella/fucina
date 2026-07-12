# Video scripts — "Forging Deep Learning in Zig"

One script per course chapter, each for a **3:00 video**. These are the
writing layer of a two-stage pipeline: a production agent takes each script
and turns it into the actual video (recordings, renders, voice, edit). A
script is therefore a *contract*: the narration is final-draft prose meant
to be spoken as written, the visuals are concrete and recordable, and the
production notes say what may bend and what must not.

## Format contract (every script follows it)

```markdown
# Video NN — Title (3:00)

*Series: Forging Deep Learning in Zig · Source: ../NN-chapter.md*

## Logline
One or two sentences: what the video teaches and what it shows off.

## Takeaways
The 2–3 things a viewer should remember. (The script highlights only the
chapter's most important ideas — it does not compress the whole chapter.)

## Script

### [m:ss–m:ss] Segment title
**VO:** The exact narration, written to be spoken.
**Visual:** What is on screen, concretely: a code excerpt (with repo path),
a terminal recording (with the exact command), a diagram (described), a
live demo (described).
**Overlay:** Short on-screen text/captions, if any.

(...segments, contiguous, ending at 3:00...)

## Asset list
Everything the production agent must record, render, or fetch: repo files
and line ranges to show, commands to run on camera, diagrams to draw,
external downloads (models) with their source.

## Production notes
Tone and pacing; what to trim first if the cut runs long; caveats that MUST
stay attached to numbers; anything that must not be changed in refinement.
```

## Hard rules (inherited from the course, they bind the videos too)

- **Numbers are quoted, never invented** — each figure carries its source
  and, for benchmarks, its dated/machine-specific framing. The caveat
  travels with the number into the video: if the VO says "2.3×", the VO or
  overlay says what the honest range is.
- **Code on screen is real** — from the repo or from the chapter's
  compile-checked course code, never mocked up. Terminal commands are the
  repo's own, runnable as shown.
- **No state-of-the-art claims; no production-readiness claims.** The
  series carries the same candor as the course (chapter 00, §0.7),
  including, where it fits, that Fucina is one person's effort with strong
  agentic-coding assistance, and that the course itself was generated with
  AI over the library as it stands today.
- **Voice** — warm, direct, concrete; motivate before formalizing; no hype.
  A video highlights the chapter's *most important* ideas — one core
  learning idea plus one capability showcase — and sends the viewer to the
  chapter for the rest.

## Timing and length

3:00 at a spoken pace of ~140–150 words/minute means **380–450 words of
VO total**. Segments are timecoded and contiguous; the final segment ends
with a one-line teaser for the next video.

## Episode index

| # | Video | Source chapter |
|---|-------|----------------|
| 00 | [Why deep learning in Zig?](00-introduction.md) | [Introduction](../00-introduction.md) |
| 01 | [Just enough Zig](01-just-enough-zig.md) | [Just enough Zig](../01-just-enough-zig.md) |
| 02 | [Just enough machine learning](02-just-enough-ml.md) | [Just enough ML](../02-just-enough-ml.md) |
| 03 | [Tensors from scratch](03-tensors-from-scratch.md) | [Tensors from scratch](../03-tensors-from-scratch.md) |
| 04 | [Axes with names](04-axes-with-names.md) | [Axes with names](../04-axes-with-names.md) |
| 05 | [The operation library](05-the-operation-library.md) | [The operation library](../05-the-operation-library.md) |
| 06 | [Going fast on CPUs](06-going-fast-on-cpus.md) | [Going fast on CPUs](../06-going-fast-on-cpus.md) |
| 07 | [Autograd: the graph hidden in the values](07-autograd.md) | [Autograd](../07-autograd.md) |
| 08 | [Training: making the machine learn](08-training.md) | [Training](../08-training.md) |
| 09 | [Training without gradients](09-training-without-gradients.md) | [Training without gradients](../09-training-without-gradients.md) |
| 10 | [The guitar amp](10-the-guitar-amp.md) | [The guitar amp](../10-the-guitar-amp.md) |
| 11 | [Model files and quantization](11-model-files-and-quantization.md) | [Model files and quantization](../11-model-files-and-quantization.md) |
| 12 | [A transformer from scratch](12-a-transformer-from-scratch.md) | [A transformer from scratch](../12-a-transformer-from-scratch.md) |
| 13 | [Inference tricks](13-inference-tricks.md) | [Inference tricks](../13-inference-tricks.md) |
| 14 | [The low-bit frontier](14-the-low-bit-frontier.md) | [The low-bit frontier](../14-the-low-bit-frontier.md) |
| 15 | [Training LLMs on your CPU](15-training-llms-on-cpu.md) | [Training LLMs on your CPU](../15-training-llms-on-cpu.md) |
| 16 | [The craft](16-the-craft.md) | [The craft](../16-the-craft.md) |
| 17 | [Your forge](17-epilogue.md) | [Epilogue](../17-epilogue.md) |
