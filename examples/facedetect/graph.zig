//! Static replay of a GGUF-embedded ONNX node list (the interpreter-driven nets:
//! anti-spoof MiniFASNet, landmarks). This is an APP-LEVEL replay confined to
//! this example — NOT a Fucina-core ONNX runtime.
//! It parses the `op;out;in;attrs` node specs and dispatches a fixed op-set over
//! raw `ExecContext` ops, mirroring the C++ `build_node`. Feature maps are
//! channel-last `[h,w,c]`; per-channel params ride zero-stride broadcast views
//! into the vectorized tail-broadcast kernels (never materialized); pooled/FC
//! tensors are rank-1/2. Weights resolve as `<prefix><name>` in the GGUF.
//!
//! The graph COMPILES once (`Compiled.compile`): specs parse, weights dequant +
//! repack, BatchNorms fold to per-channel scale/shift — then `run` is pure
//! compute per input (mirrors the reference's load-once graph lifecycle).
//! Node name slices borrow the GGUF metadata bytes, so `file` must outlive the
//! compiled graph.

const std = @import("std");
const fucina = @import("fucina");
const rec = @import("recognizer.zig");

const gguf = fucina.gguf;
const ExecContext = fucina.ExecContext;
const RawTensor = fucina.internal.RawTensor;

// --- raw weight loaders ----------------------------------------------------

fn loadConvWRaw(ctx: *ExecContext, allocator: std.mem.Allocator, file: *const gguf.File, name: []const u8) !RawTensor {
    const t = try rec.info(file, name);
    const raw = try rec.toF32(allocator, t); // ggml [kw,kh,cin/g,cout]
    defer allocator.free(raw);
    const kw = t.dims[0];
    const kh = t.dims[1];
    const cin = t.dims[2];
    const cout = t.dims[3];
    const buf = try allocator.alloc(f32, cout * kh * kw * cin);
    defer allocator.free(buf);
    for (0..cout) |co| for (0..kh) |y| for (0..kw) |x| for (0..cin) |ci| {
        buf[((co * kh + y) * kw + x) * cin + ci] = raw[((co * cin + ci) * kh + y) * kw + x];
    };
    return ctx.fromSlice(&.{ cout, kh, kw, cin }, buf);
}

fn loadVecRaw(ctx: *ExecContext, allocator: std.mem.Allocator, file: *const gguf.File, name: []const u8) !RawTensor {
    const data = try rec.toF32(allocator, try rec.info(file, name));
    defer allocator.free(data);
    return ctx.fromSlice(&.{data.len}, data);
}

/// MatMul weight: ggml ne `[out,in]` (reversed ONNX `[in,out]`) → raw `[in,out]`.
fn loadMatmulWRaw(ctx: *ExecContext, allocator: std.mem.Allocator, file: *const gguf.File, name: []const u8) !RawTensor {
    const t = try rec.info(file, name);
    const data = try rec.toF32(allocator, t);
    defer allocator.free(data);
    return ctx.fromSlice(&.{ t.dims[1], t.dims[0] }, data);
}

/// Gemm(transB=1) weight: stored ONNX B `[out,in]` → ggml ne `[in,out]`. The mul
/// needs Bᵀ `[in,out]` with `W[i,o] = B[o,i] = ggml_data[i + in·o]` (row-major).
fn loadGemmWRaw(ctx: *ExecContext, allocator: std.mem.Allocator, file: *const gguf.File, name: []const u8) !RawTensor {
    const t = try rec.info(file, name);
    const data = try rec.toF32(allocator, t);
    defer allocator.free(data);
    const in = t.dims[0];
    const out = t.dims[1];
    const buf = try allocator.alloc(f32, in * out);
    defer allocator.free(buf);
    for (0..in) |i| for (0..out) |o| {
        buf[i * out + o] = data[i + in * o];
    };
    return ctx.fromSlice(&.{ in, out }, buf);
}

// --- node parsing ----------------------------------------------------------

const max_ins = 8;
const Node = struct {
    op: []const u8,
    out: []const u8,
    ins: [max_ins][]const u8,
    n_in: usize,
    stride: usize = 1,
    pad: usize = 0,
    group: usize = 1,
    kernel: usize = 0,
    trans_b: bool = false,
    eps: f32 = -1,
};

fn parseNode(spec: []const u8) !Node {
    var it = std.mem.splitScalar(u8, spec, ';');
    const op = it.next() orelse return error.BadNode;
    const out = it.next() orelse return error.BadNode;
    const ins_csv = it.next() orelse "";
    const attrs_csv = it.next() orelse "";

    var n = Node{ .op = op, .out = out, .ins = undefined, .n_in = 0 };
    var iit = std.mem.splitScalar(u8, ins_csv, ',');
    while (iit.next()) |name| {
        if (name.len == 0) continue;
        if (n.n_in >= max_ins) return error.TooManyInputs;
        n.ins[n.n_in] = name;
        n.n_in += 1;
    }
    var ait = std.mem.splitScalar(u8, attrs_csv, ',');
    while (ait.next()) |kv| {
        const eq = std.mem.indexOfScalar(u8, kv, '=') orelse continue;
        const k = kv[0..eq];
        const v = kv[eq + 1 ..];
        if (std.mem.eql(u8, k, "s")) n.stride = try std.fmt.parseInt(usize, v, 10);
        if (std.mem.eql(u8, k, "p")) n.pad = try std.fmt.parseInt(usize, v, 10);
        if (std.mem.eql(u8, k, "g")) n.group = try std.fmt.parseInt(usize, v, 10);
        if (std.mem.eql(u8, k, "k")) n.kernel = try std.fmt.parseInt(usize, v, 10);
        if (std.mem.eql(u8, k, "tb")) n.trans_b = !std.mem.eql(u8, v, "0");
        if (std.mem.eql(u8, k, "e")) n.eps = try std.fmt.parseFloat(f32, v);
    }
    return n;
}

// --- compiled graph ---------------------------------------------------------

const BinKind = enum { add, sub, mul };

const Payload = union(enum) {
    conv: struct { w: RawTensor, b: ?RawTensor, stride: usize, pad: usize, group: usize },
    prelu: struct { alpha: RawTensor },
    relu,
    sigmoid,
    bin: struct { kind: BinKind, w: [2]?RawTensor }, // non-null = preloaded weight operand
    gap,
    flatten,
    matmul: struct { w: RawTensor, b: ?RawTensor },
    maxpool: struct { k: usize, s: usize, p: usize },
    bn: struct { scale: RawTensor, shift: RawTensor },
    copy, // Dropout / Identity

    fn deinit(self: *Payload) void {
        switch (self.*) {
            .conv => |*c| {
                c.w.deinit();
                if (c.b) |*b| b.deinit();
            },
            .prelu => |*p| p.alpha.deinit(),
            .bin => |*b| for (&b.w) |*w| {
                if (w.*) |*t| t.deinit();
            },
            .matmul => |*m| {
                m.w.deinit();
                if (m.b) |*b| b.deinit();
            },
            .bn => |*b| {
                b.scale.deinit();
                b.shift.deinit();
            },
            .relu, .sigmoid, .gap, .flatten, .maxpool, .copy => {},
        }
    }
};

const CNode = struct {
    out: []const u8,
    ins: [max_ins][]const u8,
    n_in: usize,
    payload: Payload,
};

/// Fold frozen BN stats (raw `[c]` tensors) into (scale, shift):
/// scale = γ/√(var+ε), shift = β − μ·scale, computed once at compile time.
fn bnFoldRaw(ctx: *ExecContext, g: *const RawTensor, b: *const RawTensor, m: *const RawTensor, v: *const RawTensor, eps: f32) !struct { scale: RawTensor, shift: RawTensor } {
    var ve = try ctx.addScalar(v, eps);
    defer ve.deinit();
    var sd = try ctx.sqrt(&ve);
    defer sd.deinit();
    var scale = try ctx.div(g, &sd);
    errdefer scale.deinit();
    var ms = try ctx.mul(m, &scale);
    defer ms.deinit();
    const shift = try ctx.sub(b, &ms);
    return .{ .scale = scale, .shift = shift };
}

pub const Compiled = struct {
    allocator: std.mem.Allocator,
    nodes: []CNode,
    output_name: []const u8,
    input_name: []const u8,

    pub fn deinit(self: *Compiled) void {
        for (self.nodes) |*n| n.payload.deinit();
        self.allocator.free(self.nodes);
        self.* = undefined;
    }

    /// Parse the node specs and load/fold every weight once. Name slices
    /// borrow the spec strings (GGUF metadata bytes) — `file` must outlive.
    pub fn compile(ctx: *ExecContext, allocator: std.mem.Allocator, file: *const gguf.File, prefix: []const u8, node_specs: []const []const u8, output_name: []const u8, input_name: []const u8, bn_eps: f32) !Compiled {
        // Activation names = the input + every node output; anything else fed
        // to a binary op is a weight, preloadable at compile time.
        var acts = std.StringHashMap(void).init(allocator);
        defer acts.deinit();
        try acts.put(input_name, {});
        for (node_specs) |spec| {
            const n = try parseNode(spec);
            try acts.put(n.out, {});
        }

        var nodes: std.ArrayList(CNode) = .empty;
        errdefer {
            for (nodes.items) |*n| n.payload.deinit();
            nodes.deinit(allocator);
        }

        for (node_specs) |spec| {
            const n = try parseNode(spec);
            const payload = try compileNode(ctx, allocator, file, prefix, n, &acts, bn_eps);
            try nodes.append(allocator, .{ .out = n.out, .ins = n.ins, .n_in = n.n_in, .payload = payload });
        }

        return .{
            .allocator = allocator,
            .nodes = try nodes.toOwnedSlice(allocator),
            .output_name = output_name,
            .input_name = input_name,
        };
    }

    fn wname(allocator: std.mem.Allocator, prefix: []const u8, name: []const u8) ![]u8 {
        return std.fmt.allocPrint(allocator, "{s}{s}", .{ prefix, name });
    }

    fn loadPrefixedVec(ctx: *ExecContext, allocator: std.mem.Allocator, file: *const gguf.File, prefix: []const u8, name: []const u8) !RawTensor {
        const full = try wname(allocator, prefix, name);
        defer allocator.free(full);
        return loadVecRaw(ctx, allocator, file, full);
    }

    fn compileNode(ctx: *ExecContext, allocator: std.mem.Allocator, file: *const gguf.File, prefix: []const u8, n: Node, acts: *const std.StringHashMap(void), bn_eps: f32) !Payload {
        const op = n.op;
        if (std.mem.eql(u8, op, "Conv")) {
            const full = try wname(allocator, prefix, n.ins[1]);
            defer allocator.free(full);
            var w = try loadConvWRaw(ctx, allocator, file, full);
            errdefer w.deinit();
            const b: ?RawTensor = if (n.n_in > 2) try loadPrefixedVec(ctx, allocator, file, prefix, n.ins[2]) else null;
            return .{ .conv = .{ .w = w, .b = b, .stride = n.stride, .pad = n.pad, .group = n.group } };
        } else if (std.mem.eql(u8, op, "PRelu")) {
            return .{ .prelu = .{ .alpha = try loadPrefixedVec(ctx, allocator, file, prefix, n.ins[1]) } };
        } else if (std.mem.eql(u8, op, "Relu")) {
            return .relu;
        } else if (std.mem.eql(u8, op, "Sigmoid")) {
            return .sigmoid;
        } else if (std.mem.eql(u8, op, "Add") or std.mem.eql(u8, op, "Sub") or std.mem.eql(u8, op, "Mul")) {
            const kind: BinKind = if (std.mem.eql(u8, op, "Add")) .add else if (std.mem.eql(u8, op, "Sub")) .sub else .mul;
            var w: [2]?RawTensor = .{ null, null };
            errdefer for (&w) |*t| {
                if (t.*) |*tt| tt.deinit();
            };
            for (0..2) |i| {
                if (!acts.contains(n.ins[i])) {
                    w[i] = try loadPrefixedVec(ctx, allocator, file, prefix, n.ins[i]);
                }
            }
            return .{ .bin = .{ .kind = kind, .w = w } };
        } else if (std.mem.eql(u8, op, "GlobalAveragePool")) {
            return .gap;
        } else if (std.mem.eql(u8, op, "Flatten")) {
            return .flatten;
        } else if (std.mem.eql(u8, op, "MatMul") or std.mem.eql(u8, op, "Gemm")) {
            const full = try wname(allocator, prefix, n.ins[1]);
            defer allocator.free(full);
            var w = if (std.mem.eql(u8, op, "Gemm") and n.trans_b)
                try loadGemmWRaw(ctx, allocator, file, full)
            else
                try loadMatmulWRaw(ctx, allocator, file, full);
            errdefer w.deinit();
            const b: ?RawTensor = if (n.n_in > 2) try loadPrefixedVec(ctx, allocator, file, prefix, n.ins[2]) else null;
            return .{ .matmul = .{ .w = w, .b = b } };
        } else if (std.mem.eql(u8, op, "MaxPool")) {
            return .{ .maxpool = .{ .k = n.kernel, .s = n.stride, .p = n.pad } };
        } else if (std.mem.eql(u8, op, "BatchNormalization")) {
            const eps = if (n.eps >= 0) n.eps else bn_eps;
            var g = try loadPrefixedVec(ctx, allocator, file, prefix, n.ins[1]);
            defer g.deinit();
            var b = try loadPrefixedVec(ctx, allocator, file, prefix, n.ins[2]);
            defer b.deinit();
            var m = try loadPrefixedVec(ctx, allocator, file, prefix, n.ins[3]);
            defer m.deinit();
            var v = try loadPrefixedVec(ctx, allocator, file, prefix, n.ins[4]);
            defer v.deinit();
            const ss = try bnFoldRaw(ctx, &g, &b, &m, &v, eps);
            return .{ .bn = .{ .scale = ss.scale, .shift = ss.shift } };
        } else if (std.mem.eql(u8, op, "Dropout") or std.mem.eql(u8, op, "Identity")) {
            return .copy;
        }
        return error.UnsupportedOp;
    }

    /// Replay the compiled graph on one input; returns the named output tensor
    /// (caller deinits). Pure compute — no GGUF access.
    pub fn run(self: *const Compiled, ctx: *ExecContext, allocator: std.mem.Allocator, input: *const RawTensor) !RawTensor {
        var vals = std.StringHashMap(RawTensor).init(allocator);
        defer {
            var it = vals.iterator();
            while (it.next()) |e| e.value_ptr.deinit();
            vals.deinit();
        }
        try vals.put(self.input_name, try input.cloneView());

        for (self.nodes) |*n| {
            const y = try dispatch(ctx, allocator, n, &vals);
            if (vals.fetchPut(n.out, y) catch |e| {
                var yy = y;
                yy.deinit();
                return e;
            }) |old| {
                var ov = old.value;
                ov.deinit();
            }
        }
        const out = vals.getPtr(self.output_name) orelse return error.MissingOutput;
        return out.cloneView();
    }

    fn act(vals: *std.StringHashMap(RawTensor), name: []const u8) !*RawTensor {
        return vals.getPtr(name) orelse error.MissingActivation;
    }

    fn applyBin(ctx: *ExecContext, kind: BinKind, a: *const RawTensor, b: *const RawTensor) !RawTensor {
        return switch (kind) {
            .add => ctx.add(a, b),
            .sub => ctx.sub(a, b),
            .mul => ctx.mul(a, b),
        };
    }

    /// Elementwise Add/Sub/Mul with the smaller operand (scalar / per-channel
    /// gate / weight vector) riding a zero-stride broadcast view — the exec
    /// tail-broadcast kernel consumes it directly, nothing is materialized.
    fn binaryRun(ctx: *ExecContext, kind: BinKind, a: *const RawTensor, b: *const RawTensor) !RawTensor {
        if (a.len() == b.len()) return applyBin(ctx, kind, a, b);
        if (a.len() > b.len()) {
            var bb = try b.broadcastTo(a.shape.slice());
            defer bb.deinit();
            return applyBin(ctx, kind, a, &bb);
        }
        var aa = try a.broadcastTo(b.shape.slice());
        defer aa.deinit();
        return applyBin(ctx, kind, &aa, b);
    }

    /// Global-average-pool a `[h,w,c]` map to `[1,1,c]` (keepdim so it feeds
    /// either an SE 1×1 conv or a Flatten uniformly). Two suffix-streaming
    /// reduces, one contiguous pass each — accumulation order and 1/n scaling
    /// are identical to sequential per-axis means (over h, then w).
    fn globalAvgPool(ctx: *ExecContext, x: *const RawTensor) !RawTensor {
        const sh = x.shape.slice();
        const h = sh[0];
        const w = sh[1];
        const c = sh[2];
        var s1 = try ctx.reduceBroadcast(x, &.{ w, c });
        defer s1.deinit();
        var m1 = try ctx.scale(&s1, 1.0 / @as(f32, @floatFromInt(h)));
        defer m1.deinit();
        var s2 = try ctx.reduceBroadcast(&m1, &.{c});
        defer s2.deinit();
        var m2 = try ctx.scale(&s2, 1.0 / @as(f32, @floatFromInt(w)));
        defer m2.deinit();
        return m2.reshape(&.{ 1, 1, c });
    }

    fn dispatch(ctx: *ExecContext, allocator: std.mem.Allocator, n: *const CNode, vals: *std.StringHashMap(RawTensor)) !RawTensor {
        switch (n.payload) {
            .conv => |*p| {
                const x = try act(vals, n.ins[0]);
                return ctx.conv2d(x, &p.w, if (p.b) |*b| b else null, .{ p.stride, p.stride }, .{ p.pad, p.pad }, p.group);
            },
            .prelu => |*p| {
                return ctx.preluChannels(try act(vals, n.ins[0]), &p.alpha);
            },
            .relu => return ctx.relu(try act(vals, n.ins[0])),
            .sigmoid => return ctx.sigmoid(try act(vals, n.ins[0])),
            .bin => |*p| {
                const a: *const RawTensor = if (p.w[0]) |*w| w else try act(vals, n.ins[0]);
                const b: *const RawTensor = if (p.w[1]) |*w| w else try act(vals, n.ins[1]);
                return binaryRun(ctx, p.kind, a, b);
            },
            .gap => return globalAvgPool(ctx, try act(vals, n.ins[0])),
            .flatten => {
                const x = try act(vals, n.ins[0]);
                if (x.rank() != 3) return x.reshape(&.{ 1, x.len() });
                // Reorder channel-last [h,w,c] to the reference's NCHW flatten
                // (c-major, w-fastest: idx = c·H·W + h·W + w) so the FC/Gemm
                // weight rows align.
                const sh = x.shape.slice();
                const h = sh[0];
                const w = sh[1];
                const c = sh[2];
                const xd = x.dataConst();
                const buf = try allocator.alloc(f32, h * w * c);
                defer allocator.free(buf);
                for (0..h) |yy| for (0..w) |xx| for (0..c) |cc| {
                    buf[cc * (h * w) + yy * w + xx] = xd[(yy * w + xx) * c + cc];
                };
                return ctx.fromSlice(&.{ 1, h * w * c }, buf);
            },
            .matmul => |*p| {
                const x = try act(vals, n.ins[0]);
                var xr = try x.reshape(&.{ 1, x.len() }); // [1,in]
                defer xr.deinit();
                var y = try ctx.matmul(&xr, &p.w); // [1,out]
                if (p.b) |*b| {
                    errdefer y.deinit();
                    try ctx.addAxisVectorInPlaceRank(2, &y, b.dataConst(), 1);
                }
                return y;
            },
            .maxpool => |p| {
                return ctx.maxPool2d(try act(vals, n.ins[0]), .{ p.k, p.k }, .{ p.s, p.s }, .{ p.p, p.p });
            },
            .bn => |*p| {
                return ctx.channelAffine(try act(vals, n.ins[0]), &p.scale, &p.shift);
            },
            .copy => return (try act(vals, n.ins[0])).cloneView(),
        }
    }
};
