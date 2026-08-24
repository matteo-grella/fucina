//! Architecture registry: one comptime table from a GGUF's
//! `general.architecture` string to the family module that loads and
//! decodes it. Each `Family` (declared in the family's `model.zig`)
//! exposes `Model`, the tokenizer module `Tok` (+ `Tokenizer`),
//! `load(ctx, file, options)`, `tokenizer(allocator, file)`, and
//! `template_fallback`. `serving.open` dispatches over this table with one
//! `inline for`; `familyFor` is the comptime lookup.

const std = @import("std");
const model_common = @import("model_common.zig");
const qwen3_model = @import("qwen3/model.zig");
const gemma_model = @import("gemma/model.zig");
const qwen35_model = @import("qwen35/model.zig");
const inkling_model = @import("inkling/model.zig");
const deepseek2_model = @import("deepseek2/model.zig");
const deepseek4_model = @import("deepseek4/model.zig");
const glm4moe_model = @import("glm4moe/model.zig");

/// The family-independent load levers `Family.load` accepts.
pub const LoadOptions = model_common.FamilyLoadOptions;

pub const Entry = struct {
    /// The GGUF `general.architecture` string.
    arch: []const u8,
    Family: type,
};

/// Every registered architecture. Two archs may share one family (qwen3
/// and qwen3moe drive the same descriptor runner; qwen35 and qwen35moe the
/// same hybrid stack). deepseek2 and glm4moe are registered for the
/// comptime lookup but carry no serving adapter; kimi3 stays outside the
/// decoder contract and the registry.
pub const families = [_]Entry{
    .{ .arch = "qwen3", .Family = qwen3_model.Family },
    .{ .arch = "qwen3moe", .Family = qwen3_model.Family },
    .{ .arch = "gemma4", .Family = gemma_model.Family },
    .{ .arch = "qwen35", .Family = qwen35_model.Family },
    .{ .arch = "qwen35moe", .Family = qwen35_model.Family },
    .{ .arch = "inkling", .Family = inkling_model.Family },
    .{ .arch = "deepseek2", .Family = deepseek2_model.Family },
    .{ .arch = "deepseek4", .Family = deepseek4_model.Family },
    .{ .arch = "glm4moe", .Family = glm4moe_model.Family },
};

/// Comptime lookup: the family module for an architecture string, or null
/// when the arch is not registered.
pub fn familyFor(comptime arch: []const u8) ?type {
    inline for (families) |entry| {
        if (comptime std.mem.eql(u8, entry.arch, arch)) return entry.Family;
    }
    return null;
}

test "familyFor: registered archs resolve, unknown archs return null" {
    comptime {
        std.debug.assert(familyFor("qwen3").? == qwen3_model.Family);
        std.debug.assert(familyFor("qwen3moe").? == qwen3_model.Family);
        std.debug.assert(familyFor("gemma4").? == gemma_model.Family);
        std.debug.assert(familyFor("qwen35").? == qwen35_model.Family);
        std.debug.assert(familyFor("deepseek4").? == deepseek4_model.Family);
        std.debug.assert(familyFor("llama") == null);
    }
}

test "every registered Family satisfies the decoder contract" {
    const decoder = @import("decoder.zig");
    comptime {
        for (families) |entry| decoder.assertDecoder(entry.Family.Model);
    }
}
