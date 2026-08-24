//! GGUF loader for the face-detect.cpp buffalo_l pack + the landmarks pack.
//!
//! The runtime path is **tensor-only**: it classifies every tensor by its
//! sub-model prefix and resolves weights by name. It deliberately does NOT read
//! the `facedetect.*.graph` KVs — those are the reference's ONNX-interpreter
//! spec for the anti-spoof/landmark heads; Fucina hardcodes the transcribed Zig
//! forwards, so nothing walks a node list at runtime.

const std = @import("std");
const fucina = @import("fucina");
const gguf = fucina.gguf;

/// buffalo_l / landmarks sub-model namespaces (GGUF tensor-name prefixes).
pub const Submodel = enum {
    det, // "det."   SCRFD det_10g detector   (hand-mapped, no graph KV)
    rec, // "rec."   ArcFace R50 recognizer   (hand-mapped, no graph KV)
    ga, // "ga."    genderage                (hand-mapped, no graph KV)
    antispoof, // "as<i>." MiniFASNet ensemble      (interpreter → transcribed)
    landmark2d, // "l2d."   2d106det                 (interpreter → transcribed)
    landmark3d, // "l3d."   1k3d68                   (interpreter → transcribed)
    other, // unclassified — a coverage failure (variant drift)

    /// Classify a tensor by its verbatim GGUF name prefix.
    pub fn ofTensor(name: []const u8) Submodel {
        if (std.mem.startsWith(u8, name, "det.")) return .det;
        if (std.mem.startsWith(u8, name, "rec.")) return .rec;
        if (std.mem.startsWith(u8, name, "ga.")) return .ga;
        if (std.mem.startsWith(u8, name, "l2d.")) return .landmark2d;
        if (std.mem.startsWith(u8, name, "l3d.")) return .landmark3d;
        // "as<digits>." — the anti-spoof ensemble members (as0., as1., …).
        if (name.len >= 4 and name[0] == 'a' and name[1] == 's' and std.ascii.isDigit(name[2])) {
            var i: usize = 2;
            while (i < name.len and std.ascii.isDigit(name[i])) : (i += 1) {}
            if (i < name.len and name[i] == '.') return .antispoof;
        }
        return .other;
    }
};

/// Per-sub-model tensor tally for a loaded pack.
pub const Coverage = struct {
    total: usize,
    det: usize = 0,
    rec: usize = 0,
    ga: usize = 0,
    antispoof: usize = 0,
    landmark2d: usize = 0,
    landmark3d: usize = 0,
    other: usize = 0,

    pub fn classified(self: Coverage) usize {
        return self.det + self.rec + self.ga + self.antispoof + self.landmark2d + self.landmark3d;
    }
};

/// Tally every tensor in `file` by sub-model prefix (tensor-only; no KV reads).
pub fn coverage(file: *const gguf.File) Coverage {
    var cov = Coverage{ .total = file.tensors.len };
    for (file.tensors) |t| switch (Submodel.ofTensor(t.name)) {
        .det => cov.det += 1,
        .rec => cov.rec += 1,
        .ga => cov.ga += 1,
        .antispoof => cov.antispoof += 1,
        .landmark2d => cov.landmark2d += 1,
        .landmark3d => cov.landmark3d += 1,
        .other => cov.other += 1,
    };
    return cov;
}

/// Every tensor must resolve into a known sub-model (resolved == total) —
/// total coverage surfaces variant drift immediately.
pub fn assertFullCoverage(file: *const gguf.File) !void {
    const cov = coverage(file);
    if (cov.other != 0) return error.UnclassifiedTensors;
    if (cov.classified() != cov.total) return error.CoverageMismatch;
}

/// dtype-class check: every tensor's GGML type must map to a DType Fucina
/// understands (face-detect stores conv/BN F32 and only the big FC head
/// quantized — F16/Q8_0).
pub fn assertKnownDtypes(file: *const gguf.File) !void {
    for (file.tensors) |t| {
        if (gguf.dtypeForGgmlType(t.ggml_type) == null) return error.UnknownTensorDtype;
    }
}
