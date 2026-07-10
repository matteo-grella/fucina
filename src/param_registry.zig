//! Minimal parameter registry over the public tagged Tensor facade.
//!
//! `ParamRegistry` owns no model tensors. It borrows named f32/f16/bf16 tensors,
//! retaining refcounted storage views (type-erased over dtype) for
//! checkpointing and optimizer registration. The original tensor values and
//! their GradState must outlive the registry and any optimizer it registers into.
//!
//! Schema-stability contract: a registered NAME is a checkpoint field path and
//! is part of the on-disk schema. A strict `loadStateDict` matches stream entries
//! to destinations by exact name, so RENAMING a parameter path silently breaks
//! resuming from older checkpoints (the old name no longer matches:
//! `CheckpointUnknownName` for the stream entry, `CheckpointMissingEntry` for the
//! renamed destination). When a rename is unavoidable, do NOT loosen `strict`;
//! pass a `state_dict.LoadOptions.aliases` map (`.{ .old = "enc.w", .new =
//! "encoder.w" }`) so the old checkpoint loads into the new path while keeping
//! the one-to-one strict guarantee.
const std = @import("std");
const tensor_mod = @import("tensor.zig");
const state_dict = @import("state_dict.zig");
const ag_core = @import("ag/core.zig");

const Allocator = std.mem.Allocator;
const RawTensor = tensor_mod.Tensor; // = TensorOf(.f32); the trainable dtype
const DType = tensor_mod.DType;
const Shape = tensor_mod.Shape;
const GradState = ag_core.GradState;

const max_name_len: usize = std.math.maxInt(u16);

pub const ParamRegistry = struct {
    allocator: Allocator,
    params: std.ArrayList(ParamEntry) = .empty,

    pub fn init(allocator: Allocator) ParamRegistry {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *ParamRegistry) void {
        for (self.params.items) |*param| param.deinit(self.allocator);
        self.params.deinit(self.allocator);
        self.* = undefined;
    }

    /// Register one f32/f16/bf16 tensor under `name`. Variables (grad_state !=
    /// null) are trainable; constants are registered as frozen entries
    /// (saved/loaded, but skipped by `addParamsTo`/`zeroGrad`). `name` is copied
    /// (the registry owns it); the tensor's storage is retained dtype-erased.
    pub fn addParam(self: *ParamRegistry, name: []const u8, t: anytype) !void {
        const T = comptime validateParamPtr(@TypeOf(t));
        try self.validateNewName(name);
        if (!t.value.isContiguous()) return error.NonContiguousParam;

        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);

        // Retain a heap copy of the (typed) storage view so the entry keeps the
        // bytes alive across save/load without storing a concrete dtype.
        const Raw = @TypeOf(t.value);
        const retained = try self.allocator.create(Raw);
        errdefer self.allocator.destroy(retained);
        retained.* = try t.value.cloneView();
        errdefer retained.deinit();

        // The grad-free typed (f16/bf16) facade has no grad_state field; such
        // tensors are always frozen.
        const grad_state: ?*GradState = if (comptime @hasField(T, "grad_state")) t.grad_state else null;
        try self.params.append(self.allocator, .{
            .name = owned_name,
            .dtype = T.dtype,
            .shape = t.value.shape,
            .bytes = std.mem.sliceAsBytes(retained.data()),
            .grad_state = grad_state,
            .retained = retained,
            .release = makeRelease(Raw),
        });
    }

    /// Reflectively register every f32/f16/bf16 tensor field of `model` (a
    /// mutable pointer to a struct), naming each by its field path. Nested
    /// structs recurse with dotted names ("encoder.weight"); arrays index with
    /// dots ("layers.0.weight"); tagged unions descend into the active arm
    /// under the same prefix (see `collectNode`). Unsupported-dtype tensors,
    /// scalars, and other fields are ignored. Constant tensor fields are
    /// registered (frozen).
    pub fn collect(self: *ParamRegistry, model: anytype) !void {
        return self.collectPrefixed("", model);
    }

    /// `collect` with a leading name prefix (e.g. "model" -> "model.weight").
    pub fn collectPrefixed(self: *ParamRegistry, prefix: []const u8, model: anytype) !void {
        @setEvalBranchQuota(20000);
        const info = @typeInfo(@TypeOf(model));
        if (info != .pointer or info.pointer.is_const) {
            @compileError("ParamRegistry.collect expects a mutable pointer to a model struct");
        }
        try self.collectNode(prefix, model);
    }

    fn collectNode(self: *ParamRegistry, prefix: []const u8, ptr: anytype) !void {
        const T = @typeInfo(@TypeOf(ptr)).pointer.child;
        if (comptime isTensorFacade(T)) {
            if (comptime isStateDictDtype(T.dtype)) try self.addParam(prefix, ptr);
            return; // unsupported-dtype tensors are skipped, never descended into
        }
        switch (@typeInfo(T)) {
            .@"struct" => |s| {
                inline for (s.fields) |field| {
                    if (comptime field.is_comptime) continue;
                    if (comptime !worthVisiting(field.type)) continue;
                    const name = try self.childName(prefix, field.name);
                    defer self.allocator.free(name);
                    try self.collectNode(name, &@field(ptr.*, field.name));
                }
            },
            .array => |arr| {
                if (comptime worthVisiting(arr.child)) {
                    inline for (0..arr.len) |i| {
                        const name = try self.indexName(prefix, i);
                        defer self.allocator.free(name);
                        try self.collectNode(name, &ptr.*[i]);
                    }
                }
            },
            .pointer => |p| {
                if (comptime p.is_const) return;
                switch (p.size) {
                    .one => if (comptime worthVisiting(p.child)) try self.collectNode(prefix, ptr.*),
                    .slice => if (comptime worthVisiting(p.child)) {
                        for (ptr.*, 0..) |*item, i| {
                            const name = try self.indexName(prefix, i);
                            defer self.allocator.free(name);
                            try self.collectNode(name, item);
                        }
                    },
                    else => {},
                }
            },
            .optional => {
                if (ptr.*) |*value| try self.collectNode(prefix, value);
            },
            .@"union" => |u| {
                // Tagged unions descend into the ACTIVE arm under the same
                // prefix (no arm name in the path): exactly one arm is live,
                // so names cannot collide, and the checkpoint path stays
                // stable across storage-variant arms (e.g. an f16 vs bf16
                // `LinearWeight`). Untagged unions cannot be switched on and
                // are skipped.
                if (u.tag_type != null) {
                    switch (ptr.*) {
                        inline else => |*arm| {
                            if (comptime worthVisiting(@TypeOf(arm.*))) {
                                try self.collectNode(prefix, arm);
                            }
                        },
                    }
                }
            },
            else => {},
        }
    }

    fn childName(self: *ParamRegistry, prefix: []const u8, leaf: []const u8) ![]u8 {
        if (prefix.len == 0) return self.allocator.dupe(u8, leaf);
        return std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ prefix, leaf });
    }

    fn indexName(self: *ParamRegistry, prefix: []const u8, index: usize) ![]u8 {
        if (prefix.len == 0) return std.fmt.allocPrint(self.allocator, "{d}", .{index});
        return std.fmt.allocPrint(self.allocator, "{s}.{d}", .{ prefix, index });
    }

    pub fn parameterCount(self: *const ParamRegistry) usize {
        return self.params.items.len;
    }

    /// Borrowed view of entry `index` (registration order). `bytes` aliases
    /// the registered tensor's storage (mutable — the registry retains it);
    /// `name` is owned by the registry. Both are invalidated by `deinit`.
    /// `trainable` distinguishes variables from frozen (constant) entries.
    /// This is the seam gradient-free consumers (`es.zig`) use to drive the
    /// full entry set, frozen entries included.
    pub fn view(self: *const ParamRegistry, index: usize) ParamView {
        const param = &self.params.items[index];
        return .{
            .name = param.name,
            .dtype = param.dtype,
            .shape = param.shape,
            .bytes = param.bytes,
            .trainable = param.grad_state != null,
        };
    }

    pub fn zeroGrad(self: *ParamRegistry) void {
        for (self.params.items) |*param| {
            if (param.grad_state) |state| state.zeroGrad();
        }
    }

    /// Register the trainable params (variables) into `opt` by name. Frozen
    /// entries (grad_state == null) are skipped — they are checkpoint-only.
    pub fn addParamsTo(self: *const ParamRegistry, opt: anytype) !void {
        const ParamFacade = struct {
            value: RawTensor,
            grad_state: ?*GradState,
        };

        for (self.params.items) |*param| {
            const state = param.grad_state orelse continue;
            // The autograd is f32-only, so trainable params are always f32 and
            // the retained handle is a *RawTensor (= *TensorOf(.f32)).
            if (param.dtype != .f32) continue;
            const raw: *RawTensor = @ptrCast(@alignCast(param.retained));
            var facade = ParamFacade{
                .value = try raw.cloneView(),
                .grad_state = state,
            };
            defer facade.value.deinit();
            try opt.addParamNamed(&facade, param.name);
        }
    }

    pub fn saveStateDict(self: *const ParamRegistry, writer: *std.Io.Writer) !void {
        const entries = try self.allocator.alloc(state_dict.NamedTensor, self.params.items.len);
        defer self.allocator.free(entries);
        for (self.params.items, entries) |*param, *entry| {
            entry.* = .{
                .name = param.name,
                .dtype = param.dtype,
                .shape = param.shape,
                .bytes = param.bytes,
            };
        }
        try state_dict.saveStateDict(self.allocator, writer, entries);
    }

    pub fn loadStateDict(self: *ParamRegistry, reader: *std.Io.Reader, options: state_dict.LoadOptions) !void {
        const entries = try self.allocator.alloc(state_dict.NamedTensorMut, self.params.items.len);
        defer self.allocator.free(entries);
        for (self.params.items, entries) |*param, *entry| {
            entry.* = .{
                .name = param.name,
                .dtype = param.dtype,
                .shape = param.shape,
                .bytes = param.bytes,
            };
        }
        try state_dict.loadStateDict(self.allocator, reader, entries, options);
    }

    fn validateNewName(self: *const ParamRegistry, name: []const u8) !void {
        if (name.len == 0 or name.len > max_name_len) return state_dict.Error.CheckpointInvalidName;
        if (std.mem.indexOfScalar(u8, name, 0) != null) return state_dict.Error.CheckpointInvalidName;
        if (!std.unicode.utf8ValidateSlice(name)) return state_dict.Error.CheckpointInvalidName;
        for (self.params.items) |*param| {
            if (std.mem.eql(u8, param.name, name)) return state_dict.Error.CheckpointDuplicateName;
        }
    }
};

/// Borrowed per-entry view returned by `ParamRegistry.view`.
pub const ParamView = struct {
    name: []const u8,
    dtype: DType,
    shape: Shape,
    bytes: []u8,
    trainable: bool,
};

const ParamEntry = struct {
    name: []const u8, // owned by the registry
    dtype: DType, // f32 / f16 / bf16
    shape: Shape,
    bytes: []u8, // mutable byte view aliasing the retained storage
    grad_state: ?*GradState, // null = frozen (constant): checkpoint-only
    retained: *anyopaque, // *TensorOf(dtype), heap-allocated; keeps `bytes` alive
    release: *const fn (*anyopaque, Allocator) void,

    fn deinit(self: *ParamEntry, allocator: Allocator) void {
        self.release(self.retained, allocator);
        allocator.free(self.name);
        self.* = undefined;
    }
};

/// Build the dtype-specific releaser for a retained `*TensorOf(dtype)` handle.
fn makeRelease(comptime Raw: type) *const fn (*anyopaque, Allocator) void {
    return struct {
        fn release(ptr: *anyopaque, allocator: Allocator) void {
            const v: *Raw = @ptrCast(@alignCast(ptr));
            v.deinit();
            allocator.destroy(v);
        }
    }.release;
}

/// The dtypes a ParamRegistry entry can hold (matches the safetensors state-dict codec).
fn isStateDictDtype(comptime dt: DType) bool {
    return dt == .f32 or dt == .f16 or dt == .bf16;
}

fn validateParamPtr(comptime P: type) type {
    const info = @typeInfo(P);
    if (info != .pointer or info.pointer.is_const) {
        @compileError("ParamRegistry.addParam expects a mutable pointer to an autograd Tensor");
    }
    const T = info.pointer.child;
    if (@typeInfo(T) != .@"struct" or !@hasDecl(T, "dtype") or !@hasField(T, "value") or !isStateDictDtype(T.dtype)) {
        @compileError("ParamRegistry.addParam expects a mutable pointer to an f32/f16/bf16 autograd Tensor");
    }
    return T;
}

/// A tensor facade is a struct exposing a `dtype` decl and raw `value` field.
/// `collect` treats these as leaves (never descends into their internals) and
/// registers the f32/f16/bf16 ones.
fn isTensorFacade(comptime T: type) bool {
    if (@typeInfo(T) != .@"struct") return false;
    if (!@hasDecl(T, "dtype")) return false;
    if (!@hasField(T, "value")) return false; // both the f32 and the (grad-free) typed facade have `value`
    return true;
}

/// Worth descending into during `collect`: a tensor facade (leaf), or a
/// container that can contain one. Pointers are followed only when mutable and
/// their child type is worth visiting, so fields like `*ExecContext` and
/// `*const Config` don't become accidental graph walks.
fn worthVisiting(comptime T: type) bool {
    return worthVisitingDepth(T, 0);
}

fn worthVisitingDepth(comptime T: type, comptime depth: usize) bool {
    if (depth > 8) return false;
    if (isTensorFacade(T)) return true;
    return switch (@typeInfo(T)) {
        .@"struct" => |s| blk: {
            inline for (s.fields) |field| {
                if (comptime field.is_comptime) continue;
                if (worthVisitingDepth(field.type, depth + 1)) break :blk true;
            }
            break :blk false;
        },
        .array => |arr| worthVisitingDepth(arr.child, depth + 1),
        .pointer => |p| switch (p.size) {
            .one, .slice => !p.is_const and worthVisitingDepth(p.child, depth + 1),
            else => false,
        },
        .optional => |opt| worthVisitingDepth(opt.child, depth + 1),
        .@"union" => |u| blk: {
            if (u.tag_type == null) break :blk false;
            inline for (u.fields) |field| {
                if (worthVisitingDepth(field.type, depth + 1)) break :blk true;
            }
            break :blk false;
        },
        else => false,
    };
}

test {
    _ = @import("param_registry_tests.zig");
}
