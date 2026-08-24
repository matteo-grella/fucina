//! Architecture registry: one comptime table from a GGUF's
//! `general.architecture` string to the family module that loads and
//! decodes it. Each `Family` (declared in the family's `model.zig`)
//! exposes `Model`, the tokenizer module `Tok` (+ `Tokenizer`),
//! `load(ctx, file, options)`, `tokenizer(allocator, file)`, and
//! `template_fallback`; a served row also names its serving wiring
//! (`Entry.Serving`, the family's `serving.zig`). `serving.open`
//! dispatches over this table with one `inline for`; `familyFor` is the
//! comptime lookup.

const std = @import("std");
const model_common = @import("model_common.zig");
const qwen3_model = @import("qwen3/model.zig");
const qwen3_serving = @import("qwen3/serving.zig");
const gemma_model = @import("gemma/model.zig");
const gemma_serving = @import("gemma/serving.zig");
const qwen35_model = @import("qwen35/model.zig");
const qwen35_serving = @import("qwen35/serving.zig");
const inkling_model = @import("inkling/model.zig");
const inkling_serving = @import("inkling/serving.zig");
const deepseek2_model = @import("deepseek2/model.zig");
const deepseek4_model = @import("deepseek4/model.zig");
const deepseek4_serving = @import("deepseek4/serving.zig");
const glm4moe_model = @import("glm4moe/model.zig");

/// The family-independent load levers `Family.load` accepts.
pub const LoadOptions = model_common.FamilyLoadOptions;

pub const Entry = struct {
    /// The GGUF `general.architecture` string.
    arch: []const u8,
    Family: type,
    /// The family's serving wiring (its `serving.zig`: `openFromFile` plus
    /// the `conversation_hosted` flag), or null when the family is
    /// registered for the comptime lookup only.
    Serving: ?type = null,
};

/// Every registered architecture. Two archs may share one family (qwen3
/// and qwen3moe drive the same descriptor runner; qwen35 and qwen35moe the
/// same hybrid stack); kimi3 stays outside the decoder contract and the
/// registry. deepseek2 and glm4moe carry no serving wiring: deepseek2's
/// MLA cache has no rewind (`caps.rewind = false`, so no `Conversation`)
/// and its GGUFs carry no recognized chat template; glm4moe decodes
/// through a mutable model (`forwardStep` takes `*Model`) while the
/// `Conversation`/`GgufChatBackend` engine stores `*const Model`, and it
/// has no recognized template either.
pub const families = [_]Entry{
    .{ .arch = "qwen3", .Family = qwen3_model.Family, .Serving = qwen3_serving },
    .{ .arch = "qwen3moe", .Family = qwen3_model.Family, .Serving = qwen3_serving },
    .{ .arch = "gemma4", .Family = gemma_model.Family, .Serving = gemma_serving },
    .{ .arch = "qwen35", .Family = qwen35_model.Family, .Serving = qwen35_serving },
    .{ .arch = "qwen35moe", .Family = qwen35_model.Family, .Serving = qwen35_serving },
    .{ .arch = "inkling", .Family = inkling_model.Family, .Serving = inkling_serving },
    .{ .arch = "deepseek2", .Family = deepseek2_model.Family },
    // The deepseek2 family's Config dispatches metadata prefixes for all
    // three archs (deepseek2/model.zig), so the GLM-DSA and DeepSeek-3.2
    // variants are registry-reachable (fucina-run completion/NLL/parity),
    // serving excluded like the base arch.
    .{ .arch = "glm-dsa", .Family = deepseek2_model.Family },
    .{ .arch = "deepseek32", .Family = deepseek2_model.Family },
    .{ .arch = "deepseek4", .Family = deepseek4_model.Family, .Serving = deepseek4_serving },
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
