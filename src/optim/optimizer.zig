//! The one optimizer body, `Optimizer(Kernel)`: registration and dedupe,
//! the fallback route, zeroGrad/norm/clip, the slot walk of `step`, and
//! the checkpoint frame (version rule, pinned config header, staged
//! transactional load). A kernel (`moment_pair`, `sgd`, `muon`, `apollo`)
//! supplies only its config, per-slot state, update, and record layout.
//! Nothing here knows a specific optimizer.

const std = @import("std");
const common = @import("common.zig");
const frame = @import("frame.zig");

const Allocator = std.mem.Allocator;
const ExecContext = common.ExecContext;
const GradState = common.GradState;
const OptimError = common.OptimError;
const StateBuf = common.StateBuf;
const Param = common.Param;
const takeGrad = common.takeGrad;
const paramGradSqNorm = common.paramGradSqNorm;
const scaleParamGrad = common.scaleParamGrad;
const clipByGlobalNorm = common.clipByGlobalNorm;
const GradStateSet = common.GradStateSet;
const gradStatesCollide = common.gradStatesCollide;
const insertGradStates = common.insertGradStates;

// Transactional load contract: every `loadState` (and `frame.loadTensors`)
// is all-or-nothing. Each record is decoded into freshly-allocated scratch
// and the WHOLE stream is validated (magic, pinned config, names, dims,
// lengths, slot match) BEFORE any live parameter/optimizer buffer is
// written. A truncated, short, or otherwise-invalid stream therefore leaves
// every destination byte-unchanged; a half-applied checkpoint can silently
// corrupt training, so one is never produced. (`OptimizerSet.loadState` is
// transactional per member optimizer.)

/// How a kernel's fallback optimizer shares the checkpoint: `.nested`
/// writes the fallback's own complete frame (magic, header, slots) after
/// this one (Muon over AdamW); `.inline_slots` appends only the fallback's
/// slot list, sharing this frame's version and header (Apollo over its
/// legacy-HF AdamW path).
pub const FallbackFrame = enum { nested, inline_slots };

/// The one optimizer body. A `Kernel` supplies what differs between
/// optimizers and nothing else:
///
/// - `Config`: the hyperparameter struct `init` stores.
/// - `State`: the per-slot buffers and counters, with
///   `deinit(*State, Allocator)`.
/// - `initState(allocator, config, param: *const Param, index) !State`:
///   allocate one slot's state (`index` is its position in the slot list).
/// - `update(ctx, config, param: *Param, state: *State, grad: *const RawTensor) !void`:
///   one slot's step over the param's f32 data (`param.data()`).
/// - `magics: frame.Magics`: the frame's v3/v4/v5 magics.
/// - `pinned_config_fields: []const []const u8`: the structural `Config`
///   fields written into the frame header and required to match on load
///   (`error.CheckpointConfigMismatch`); `lr` is never one.
/// - `record: []const []const u8`: the `State` fields serialized per slot,
///   in wire order. The encoding follows each field's type
///   (`frame.writeRecordField`): u64/f32 scalars, dtype-tagged `StateBuf`s,
///   raw `[]f32`s. Fields outside `record` are transient.
/// - `checkConfig(config) !void` (optional): constructor-time validation
///   (`error.InvalidOptimizerConfig`).
/// - `afterLoad(state: *State) void` (optional): per-slot hook after a
///   committed load.
/// - `Fallback` (optional): the `Optimizer(...)` type receiving the params
///   `routesToFallback(param: *const Param) bool` sends there and the
///   `addFallbackParam*` registrations (which degrade to `addParam*` when
///   no fallback exists); `fallbackConfig(config)` derives
///   its config and `fallback_frame: FallbackFrame` says how it shares
///   the frame. A kernel that only ever frames inline needs no `magics`
///   or `pinned_config_fields`.
///
/// Frames: v3 iff every `StateBuf` of every slot is f32, v4 otherwise, v5
/// whenever a slot carries an f32 master (16-bit params); an inline
/// fallback's slots vote too. Wire bytes are pinned by the checkpoint
/// goldens in optim_tests.zig.
pub fn Optimizer(comptime Kernel: type) type {
    const has_fallback = @hasDecl(Kernel, "Fallback");
    return struct {
        const Self = @This();
        pub const Config = Kernel.Config;
        pub const State = Kernel.State;
        pub const Slot = struct {
            param: Param,
            state: State,
        };
        const Fallback = if (has_fallback) Kernel.Fallback else void;

        allocator: Allocator,
        config: Config,
        slots: std.ArrayList(Slot) = .empty,
        fallback: Fallback,

        comptime {
            if (@hasDecl(Kernel, "magics") and Kernel.magics.v4 == null) {
                for (Kernel.record) |name| {
                    if (@FieldType(State, name) == StateBuf) {
                        @compileError(@typeName(Kernel) ++ ": a dtype-tagged state buffer needs a v4 magic");
                    }
                }
            }
        }

        pub fn init(allocator: Allocator, config: Config) !Self {
            if (@hasDecl(Kernel, "checkConfig")) try Kernel.checkConfig(config);
            return .{
                .allocator = allocator,
                .config = config,
                .fallback = if (has_fallback) try Fallback.init(allocator, Kernel.fallbackConfig(config)) else {},
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.slots.items) |*slot| {
                slot.state.deinit(self.allocator);
                slot.param.deinit(self.allocator);
            }
            self.slots.deinit(self.allocator);
            if (has_fallback) self.fallback.deinit();
            self.* = undefined;
        }

        pub fn addParam(self: *Self, t: anytype) !void {
            var param = try Param.of(t);
            errdefer param.deinit(self.allocator);
            try self.addOwnedParam(param);
        }

        /// `addParam` plus a checkpoint name (borrowed; see `Param.name`).
        pub fn addParamNamed(self: *Self, t: anytype, name: []const u8) !void {
            var param = try Param.of(t);
            errdefer param.deinit(self.allocator);
            param.name = name;
            try self.addOwnedParam(param);
        }

        /// Force a param onto the fallback (embeddings, lm/classifier heads).
        /// Without an embedded fallback (SGD/Adam/AdamW) this IS `addParam`:
        /// the one path is exactly where a fallback-routed param lands, so
        /// generic registration code may call it unconditionally.
        pub fn addFallbackParam(self: *Self, t: anytype) !void {
            var param = try Param.of(t);
            errdefer param.deinit(self.allocator);
            try self.addOwnedFallbackParam(param);
        }

        /// `addFallbackParam` plus a checkpoint name (borrowed; see `Param.name`).
        pub fn addFallbackParamNamed(self: *Self, t: anytype, name: []const u8) !void {
            var param = try Param.of(t);
            errdefer param.deinit(self.allocator);
            param.name = name;
            try self.addOwnedFallbackParam(param);
        }

        fn addOwnedFallbackParam(self: *Self, param: Param) !void {
            if (has_fallback) {
                if (self.containsGradState(param.grad_state)) return OptimError.DuplicateParam;
                return self.fallback.addOwnedParam(param);
            }
            return self.addOwnedParam(param);
        }

        /// Internal seam (pub for the fallback route), not API: take
        /// ownership of `param`, route it, and allocate its slot state.
        pub fn addOwnedParam(self: *Self, param: Param) !void {
            if (self.containsGradState(param.grad_state)) return OptimError.DuplicateParam;
            if (has_fallback) {
                if (Kernel.routesToFallback(&param)) return self.fallback.addOwnedParam(param);
            }
            var owned = param;
            try owned.ensureMaster(self.allocator);
            errdefer if (owned.master.len != 0) self.allocator.free(owned.master);
            var state = try Kernel.initState(self.allocator, self.config, &owned, self.slots.items.len);
            errdefer state.deinit(self.allocator);
            try self.slots.append(self.allocator, .{ .param = owned, .state = state });
        }

        /// Internal seam (pub for the fallback route), not API.
        pub fn containsGradState(self: *const Self, state: *const GradState) bool {
            for (self.slots.items) |*slot| {
                if (slot.param.grad_state == state) return true;
            }
            return if (has_fallback) self.fallback.containsGradState(state) else false;
        }

        /// Add this optimizer's grad-states to `set` for `OptimizerSet`'s
        /// cross-member duplicate check; `DuplicateParam` if any already present
        /// (set left unchanged on collision).
        pub fn collectGradStates(self: *const Self, set: *GradStateSet, allocator: Allocator) !void {
            if (gradStatesCollide(set, self.slots.items)) return OptimError.DuplicateParam;
            if (has_fallback) {
                if (gradStatesCollide(set, self.fallback.slots.items)) return OptimError.DuplicateParam;
            }
            try insertGradStates(set, allocator, self.slots.items);
            if (has_fallback) try insertGradStates(set, allocator, self.fallback.slots.items);
        }

        /// One `Kernel.update` per slot with a gradient (a param without one
        /// is skipped, PyTorch-style), then the fallback's step.
        pub fn step(self: *Self, ctx: *ExecContext) !void {
            for (self.slots.items) |*slot| {
                var grad = (try takeGrad(ctx, &slot.param)) orelse continue;
                defer grad.deinit();
                try Kernel.update(ctx, self.config, &slot.param, &slot.state, &grad);
                slot.param.publish();
            }
            if (has_fallback) try self.fallback.step(ctx);
        }

        pub fn zeroGrad(self: *Self) void {
            for (self.slots.items) |*slot| slot.param.grad_state.zeroGrad();
            if (has_fallback) self.fallback.zeroGrad();
        }

        pub fn gradSquaredNorm(self: *Self, ctx: *ExecContext) !f64 {
            var total: f64 = 0;
            for (self.slots.items) |*slot| total += try paramGradSqNorm(ctx, &slot.param);
            if (has_fallback) total += try self.fallback.gradSquaredNorm(ctx);
            return total;
        }

        pub fn scaleGradients(self: *Self, ctx: *ExecContext, factor: f32) !void {
            for (self.slots.items) |*slot| try scaleParamGrad(ctx, &slot.param, factor);
            if (has_fallback) try self.fallback.scaleGradients(ctx, factor);
        }

        /// L2 global-norm clip over this optimizer's params, fallback
        /// included (after backward, before step). Returns the pre-clip norm.
        pub fn clipGradNorm(self: *Self, ctx: *ExecContext, max_norm: f32) !f32 {
            return clipByGlobalNorm(ctx, self, max_norm);
        }

        // -- Checkpoint frame --------------------------------------------

        /// The version rule (see the type doc). Pub for a nested fallback's
        /// own frame; an inline fallback's slots are folded in here.
        pub fn frameVersion(self: *const Self) frame.FrameVersion {
            var version: frame.FrameVersion = .v3;
            for (self.slots.items) |*slot| {
                inline for (Kernel.record) |name| {
                    if (@FieldType(State, name) == StateBuf) {
                        if (@field(slot.state, name) != .f32) version = .v4;
                    }
                }
            }
            if (frame.slotsCarryMasters(self.slots.items)) version = .v5;
            if (has_fallback) {
                if (Kernel.fallback_frame == .inline_slots) {
                    const fb = self.fallback.frameVersion();
                    if (@intFromEnum(fb) > @intFromEnum(version)) version = fb;
                }
            }
            return version;
        }

        pub fn saveState(self: *const Self, writer: *std.Io.Writer) !void {
            // Names are validated before the first byte is written, the
            // inline fallback's included.
            try frame.validateSlotNames(self.allocator, self.slots.items);
            if (has_fallback) {
                if (Kernel.fallback_frame == .inline_slots) try frame.validateSlotNames(self.allocator, self.fallback.slots.items);
            }
            const version = self.frameVersion();
            try writer.writeAll(Kernel.magics.of(version));
            inline for (Kernel.pinned_config_fields) |name| try frame.writeScalar(writer, @field(self.config, name));
            try self.writeSlots(writer, version);
            if (has_fallback) {
                switch (Kernel.fallback_frame) {
                    .nested => try self.fallback.saveState(writer),
                    .inline_slots => try self.fallback.writeSlots(writer, version),
                }
            }
        }

        pub fn loadState(self: *Self, reader: *std.Io.Reader) !void {
            const version = try Kernel.magics.expect(reader);
            inline for (Kernel.pinned_config_fields) |name| try frame.expectScalar(reader, @field(self.config, name));
            var staged = try self.stageSlots(reader, version);
            defer self.freeStaged(&staged);
            if (has_fallback) {
                switch (Kernel.fallback_frame) {
                    .nested => {
                        // The fallback's own frame follows and loads
                        // transactionally too; committing after it keeps the
                        // whole atomic.
                        try self.fallback.loadState(reader);
                        self.commitSlots(staged.items);
                    },
                    .inline_slots => {
                        var fallback_staged = try self.fallback.stageSlots(reader, version);
                        defer self.fallback.freeStaged(&fallback_staged);
                        self.commitSlots(staged.items);
                        self.fallback.commitSlots(fallback_staged.items);
                    },
                }
            } else {
                self.commitSlots(staged.items);
            }
        }

        /// The slot list: u32 count, then per slot name, dims, the `record`
        /// fields in order, and the v5 master.
        fn writeSlots(self: *const Self, writer: *std.Io.Writer, version: frame.FrameVersion) !void {
            try writer.writeInt(u32, @intCast(self.slots.items.len), .little);
            for (self.slots.items, 0..) |*slot, i| {
                try frame.writeSlotName(writer, &slot.param, i);
                try frame.writeSlotDims(writer, &slot.param);
                inline for (Kernel.record) |name| try frame.writeRecordField(writer, version, @field(slot.state, name));
                try frame.writeSlotMaster(writer, version, &slot.param);
            }
        }

        /// One decoded-but-uncommitted slot record: the destination slot,
        /// the record scalars decoded into a copy of its State, the state
        /// buffers' storage bytes in record order (dtype already validated,
        /// so the commit is a plain byte copy), and the v5 master when the
        /// frame carried one.
        const Staged = struct {
            idx: usize,
            state: State,
            data: []u8,
            master: []f32 = &.{},
        };

        fn stagedBytes(state: *const State) usize {
            var total: usize = 0;
            inline for (Kernel.record) |name| total += frame.recordFieldBytes(@field(state, name));
            return total;
        }

        fn stageSlots(self: *Self, reader: *std.Io.Reader, version: frame.FrameVersion) !std.ArrayList(Staged) {
            const count = try reader.takeInt(u32, .little);
            var matcher = try frame.SlotMatcher.init(self.allocator, self.slots.items);
            defer matcher.deinit();
            var staged = try std.ArrayList(Staged).initCapacity(self.allocator, count);
            errdefer self.freeStaged(&staged);
            for (0..count) |_| {
                const idx = try matcher.match(reader);
                const slot = &self.slots.items[idx];
                try frame.expectSlotDims(reader, &slot.param);
                var entry: Staged = .{ .idx = idx, .state = slot.state, .data = try self.allocator.alloc(u8, stagedBytes(&slot.state)) };
                errdefer self.allocator.free(entry.data);
                var offset: usize = 0;
                inline for (Kernel.record) |name| {
                    const F = @FieldType(State, name);
                    const value = @field(slot.state, name);
                    if (F == StateBuf) {
                        const n = value.byteLen();
                        try frame.readStateSlice(reader, version, value, entry.data[offset..][0..n]);
                        offset += n;
                    } else if (F == []f32) {
                        const n = 4 * value.len;
                        try reader.readSliceAll(entry.data[offset..][0..n]);
                        offset += n;
                    } else {
                        @field(entry.state, name) = try frame.readScalar(reader, F);
                    }
                }
                entry.master = try frame.readSlotMaster(self.allocator, reader, version, &slot.param);
                staged.appendAssumeCapacity(entry);
            }
            try matcher.requireAllFilled();
            return staged;
        }

        /// Commit fully-validated records: no failure point remains.
        fn commitSlots(self: *Self, staged: []const Staged) void {
            for (staged) |s| {
                const slot = &self.slots.items[s.idx];
                var offset: usize = 0;
                inline for (Kernel.record) |name| {
                    const F = @FieldType(State, name);
                    if (F == StateBuf) {
                        const dest = @field(slot.state, name).bytes();
                        @memcpy(dest, s.data[offset..][0..dest.len]);
                        offset += dest.len;
                    } else if (F == []f32) {
                        const dest = std.mem.sliceAsBytes(@field(slot.state, name));
                        @memcpy(dest, s.data[offset..][0..dest.len]);
                        offset += dest.len;
                    } else {
                        @field(slot.state, name) = @field(s.state, name);
                    }
                }
                frame.commitSlotMaster(&slot.param, s.master);
                if (@hasDecl(Kernel, "afterLoad")) Kernel.afterLoad(&slot.state);
            }
        }

        fn freeStaged(self: *Self, staged: *std.ArrayList(Staged)) void {
            for (staged.items) |s| {
                self.allocator.free(s.data);
                if (s.master.len != 0) self.allocator.free(s.master);
            }
            staged.deinit(self.allocator);
        }
    };
}
