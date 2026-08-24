# Fucina Reference

The detailed reference for the Fucina library: the public API surface, its
exact semantics, and the internal layers needed to extend it. Structure and
rationale live in [ARCHITECTURE.md](../ARCHITECTURE.md); this document is the
API-level companion. Every Zig snippet is machine-verified against the tree
(see [§1](01-introduction-and-mental-model.md)).

## Chapters

1. [Introduction and mental model](01-introduction-and-mental-model.md)
2. [Toolchain, build, and project wiring](02-toolchain-build-and-project-wiring.md)
3. [Tensors: types, construction, and data access](03-tensors-types-construction-and-data-access.md)
4. [Tensor operations](04-tensor-operations.md)
5. [Automatic differentiation](05-automatic-differentiation.md)
6. [The execution runtime: ExecContext and the memory model](06-the-execution-runtime-execcontext-and-the-memory-model.md)
7. [Named axes: the tag algebra](07-named-axes-the-tag-algebra.md)
8. [Data types, storage, and the raw tensor layer (internal)](08-data-types-storage-and-the-raw-tensor-layer-internal.md)
9. [Backends: CPU SIMD, BLAS, threading, and GPU offload](09-backends-cpu-simd-blas-threading-and-gpu-offload.md)
10. [Quantization](10-quantization.md)
11. [Training: optimizers, evolution strategies, LoRA, and checkpoints](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md)
12. [Model I/O: GGUF and safetensors](12-model-io-gguf-and-safetensors.md)
13. [The model stack (fucina_models)](13-the-model-stack-fucina_models.md)
14. [Model families and example applications](14-model-families-and-example-applications.md)
