//! Param groups: `AnyOptimizer` (type-erased vtable over one optimizer
//! instance) and `OptimizerSet` (several instances behind one step /
//! zeroGrad / global clip / saveState / loadState, with the
//! cross-instance duplicate-registration guard). Frame magic FZO3.

const std = @import("std");
const common = @import("common.zig");
const frame = @import("frame.zig");

const Allocator = std.mem.Allocator;
const ExecContext = common.ExecContext;
const OptimError = common.OptimError;
const clipByGlobalNorm = common.clipByGlobalNorm;
const GradStateSet = common.GradStateSet;
const expectMagic = frame.expectMagic;

// ---------------------------------------------------------------------------
// Param groups: a type-erased set of optimizer instances.
//
// A "param group" is exactly {hyperparams, params, state} — which is what one
// optimizer instance already is. OptimizerSet aggregates instances behind one
// step / zeroGrad / clipGradNorm (GLOBAL norm, like PyTorch's
// clip_grad_norm_(model.parameters())) / saveState / loadState surface.
// ---------------------------------------------------------------------------

pub const AnyOptimizer = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        step: *const fn (*anyopaque, *ExecContext) anyerror!void,
        zeroGrad: *const fn (*anyopaque) void,
        gradSquaredNorm: *const fn (*anyopaque, *ExecContext) anyerror!f64,
        scaleGradients: *const fn (*anyopaque, *ExecContext, f32) anyerror!void,
        saveState: *const fn (*anyopaque, *std.Io.Writer) anyerror!void,
        loadState: *const fn (*anyopaque, *std.Io.Reader) anyerror!void,
    };

    pub fn step(self: AnyOptimizer, ctx: *ExecContext) !void {
        return self.vtable.step(self.ptr, ctx);
    }

    pub fn zeroGrad(self: AnyOptimizer) void {
        self.vtable.zeroGrad(self.ptr);
    }

    pub fn gradSquaredNorm(self: AnyOptimizer, ctx: *ExecContext) !f64 {
        return self.vtable.gradSquaredNorm(self.ptr, ctx);
    }

    pub fn scaleGradients(self: AnyOptimizer, ctx: *ExecContext, factor: f32) !void {
        return self.vtable.scaleGradients(self.ptr, ctx, factor);
    }

    pub fn saveState(self: AnyOptimizer, writer: *std.Io.Writer) !void {
        return self.vtable.saveState(self.ptr, writer);
    }

    pub fn loadState(self: AnyOptimizer, reader: *std.Io.Reader) !void {
        return self.vtable.loadState(self.ptr, reader);
    }
};

/// Wrap a concrete optimizer pointer (Adam/AdamW/SGD/Muon/Apollo) as AnyOptimizer.
/// The wrapped optimizer is borrowed: the caller still owns and deinits it,
/// and it must not move (or be freed) while the AnyOptimizer/OptimizerSet is
/// in use — the raw pointer is captured.
pub fn anyOptimizer(opt: anytype) AnyOptimizer {
    const T = @typeInfo(@TypeOf(opt)).pointer.child;
    const Impl = struct {
        fn step(ptr: *anyopaque, ctx: *ExecContext) anyerror!void {
            const self: *T = @ptrCast(@alignCast(ptr));
            return self.step(ctx);
        }
        fn zeroGrad(ptr: *anyopaque) void {
            const self: *T = @ptrCast(@alignCast(ptr));
            self.zeroGrad();
        }
        fn gradSquaredNorm(ptr: *anyopaque, ctx: *ExecContext) anyerror!f64 {
            const self: *T = @ptrCast(@alignCast(ptr));
            return self.gradSquaredNorm(ctx);
        }
        fn scaleGradients(ptr: *anyopaque, ctx: *ExecContext, factor: f32) anyerror!void {
            const self: *T = @ptrCast(@alignCast(ptr));
            return self.scaleGradients(ctx, factor);
        }
        fn saveState(ptr: *anyopaque, writer: *std.Io.Writer) anyerror!void {
            const self: *T = @ptrCast(@alignCast(ptr));
            return self.saveState(writer);
        }
        fn loadState(ptr: *anyopaque, reader: *std.Io.Reader) anyerror!void {
            const self: *T = @ptrCast(@alignCast(ptr));
            return self.loadState(reader);
        }
        const vtable = AnyOptimizer.VTable{
            .step = step,
            .zeroGrad = zeroGrad,
            .gradSquaredNorm = gradSquaredNorm,
            .scaleGradients = scaleGradients,
            .saveState = saveState,
            .loadState = loadState,
        };
    };
    return .{ .ptr = opt, .vtable = &Impl.vtable };
}

pub const OptimizerSet = struct {
    allocator: Allocator,
    items: std.ArrayList(AnyOptimizer) = .empty,
    /// Every grad-state registered across ALL member optimizers; guards against
    /// cross-instance duplicate registration (silent double-step).
    grad_states: GradStateSet = .empty,

    pub fn init(allocator: Allocator) OptimizerSet {
        return .{ .allocator = allocator };
    }

    /// Frees only the set; the member optimizers stay owned by the caller.
    pub fn deinit(self: *OptimizerSet) void {
        self.grad_states.deinit(self.allocator);
        self.items.deinit(self.allocator);
        self.* = undefined;
    }

    /// Register a member optimizer. Returns `OptimError.DuplicateParam` if any of
    /// its parameters' grad-states is ALREADY registered with a previously-added
    /// member (the same `Variable` in two groups) — closing the cross-instance
    /// gap the per-optimizer guard leaves open. On a collision the set is
    /// unchanged and the member is NOT added.
    pub fn add(self: *OptimizerSet, opt: anytype) !void {
        try opt.collectGradStates(&self.grad_states, self.allocator);
        try self.items.append(self.allocator, anyOptimizer(opt));
    }

    pub fn step(self: *OptimizerSet, ctx: *ExecContext) !void {
        for (self.items.items) |opt| try opt.step(ctx);
    }

    pub fn zeroGrad(self: *OptimizerSet) void {
        for (self.items.items) |opt| opt.zeroGrad();
    }

    pub fn gradSquaredNorm(self: *OptimizerSet, ctx: *ExecContext) !f64 {
        var total: f64 = 0;
        for (self.items.items) |opt| total += try opt.gradSquaredNorm(ctx);
        return total;
    }

    pub fn scaleGradients(self: *OptimizerSet, ctx: *ExecContext, factor: f32) !void {
        for (self.items.items) |opt| try opt.scaleGradients(ctx, factor);
    }

    /// GLOBAL norm across every group — the PyTorch clip_grad_norm_ contract.
    pub fn clipGradNorm(self: *OptimizerSet, ctx: *ExecContext, max_norm: f32) !f32 {
        return clipByGlobalNorm(ctx, self, max_norm);
    }

    pub fn saveState(self: *const OptimizerSet, writer: *std.Io.Writer) !void {
        try writer.writeAll("FZO3");
        try writer.writeInt(u32, @intCast(self.items.items.len), .little);
        for (self.items.items) |opt| try opt.saveState(writer);
    }

    pub fn loadState(self: *OptimizerSet, reader: *std.Io.Reader) !void {
        try expectMagic(reader, "FZO3");
        const count = try reader.takeInt(u32, .little);
        if (count != self.items.items.len) return OptimError.CheckpointShapeMismatch;
        for (self.items.items) |opt| try opt.loadState(reader);
    }
};
