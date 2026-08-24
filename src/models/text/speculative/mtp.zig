//! Native MTP (multi-token-prediction) drafting behind the `DraftSource`
//! vtable: the family's `nextn`-style MTP head proposes the draft and the
//! shared `SpeculativeDecoder` verify loop drives commit/rewind, replacing
//! hand-rolled draft/verify/commit loops (examples/glm4moe). The adapter is
//! generic over a glm4moe-shaped model: `mtpDraftStep(self, ctx, mtp_cache,
//! token, h_prev, h_out)`, an `initMtpCache(self, capacity)` constructor
//! for the MTP stream's own cache, and a model-owned `step_hiddens` row
//! buffer (the trunk hiddens of the most recent forward, one row per
//! processed position). Requires `Model.caps.rewind` (the decoder truncates
//! rejected draft positions).
//!
//! deepseek4's MTP sidecar stays on its own loop: its `Session` state
//! rewinds by snapshot/restore, not `truncate`, so it cannot ride the
//! decoder (`deepseek4.Model.caps.rewind == false`).
//!
//! Bookkeeping contract (mirrors the decoder's): the adapter's `hiddens`
//! holds ONE trunk row per CACHED position. `observe(committed)` appends
//! the first `committed.len` rows of `model.step_hiddens` — exactly the
//! verify rows the decoder's truncate keeps — and `observePrefill` seeds
//! the prompt rows after the caller's own prefill forward. `suggest` first
//! catches the MTP stream up on committed positions (position `i` consumes
//! `(token[i+1], hidden[i])`), then chains `depth` drafts from the
//! frontier and rewinds the MTP cache past the catch-up point.

const std = @import("std");
const fucina = @import("fucina");
const core = @import("core.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;

pub fn MtpDraftSource(comptime Model: type) type {
    comptime {
        if (!@hasDecl(Model, "mtpDraftStep"))
            @compileError(@typeName(Model) ++ " has no `mtpDraftStep` (native MTP head required)");
        if (!@hasDecl(Model, "initMtpCache"))
            @compileError(@typeName(Model) ++ " has no `initMtpCache` (the MTP stream's cache constructor)");
        if (!Model.caps.rewind)
            @compileError(@typeName(Model) ++ ": MTP drafting rides SpeculativeDecoder, which requires caps.rewind");
    }
    return struct {
        const Self = @This();

        allocator: Allocator,
        ctx: *ExecContext,
        model: *Model,
        /// The MTP stream's own cache (positions follow the trunk's).
        mtp_cache: Model.Cache,
        /// Trunk hidden state of every cached trunk position
        /// (`[cache.len() rows, hidden]`).
        hiddens: std.ArrayList(f32),
        /// MTP positions already fed: position `i` consumed
        /// `(token[i+1], hiddens[i])`.
        mtp_fed: usize = 0,
        /// Draft chain length per round.
        depth: usize,
        h_prev: []f32,
        h_scratch: []f32,
        /// `suggest` cannot return errors through the vtable: a failed
        /// draft round lands here (the round proposes no draft; the
        /// decoder falls back to a plain step).
        err: ?anyerror = null,

        pub fn init(ctx: *ExecContext, model: *Model, capacity: usize, depth: usize) !Self {
            const allocator = ctx.allocator;
            var mtp_cache = try model.initMtpCache(capacity);
            errdefer mtp_cache.deinit();
            const hidden = model.config.hidden_size;
            const h_prev = try allocator.alloc(f32, hidden);
            errdefer allocator.free(h_prev);
            const h_scratch = try allocator.alloc(f32, hidden);
            errdefer allocator.free(h_scratch);
            return .{
                .allocator = allocator,
                .ctx = ctx,
                .model = model,
                .mtp_cache = mtp_cache,
                .hiddens = .empty,
                .depth = depth,
                .h_prev = h_prev,
                .h_scratch = h_scratch,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.h_scratch);
            self.allocator.free(self.h_prev);
            self.hiddens.deinit(self.allocator);
            self.mtp_cache.deinit();
            self.* = undefined;
        }

        /// Seed the trunk hiddens of the caller's prefill forward
        /// (`model.step_hiddens` holds one row per prompt position). Call
        /// once, after the prefill and before the first decoder step.
        pub fn observePrefill(self: *Self) !void {
            try self.hiddens.appendSlice(self.allocator, self.model.step_hiddens);
        }

        pub fn source(self: *Self) core.DraftSource {
            return .{ .ptr = self, .vtable = &vtable };
        }

        const vtable = core.DraftSource.VTable{
            .suggest = vtSuggest,
            .observe = vtObserve,
        };

        fn vtSuggest(ptr: *anyopaque, context: []const usize, buf: []usize) usize {
            const self: *Self = @ptrCast(@alignCast(ptr));
            return self.draftRound(context, buf) catch |err| {
                self.err = err;
                self.mtp_cache.truncate(self.mtp_fed);
                return 0;
            };
        }

        fn draftRound(self: *Self, context: []const usize, buf: []usize) !usize {
            const hidden = self.model.config.hidden_size;
            const cached = self.hiddens.items.len / hidden;
            if (cached == 0 or context.len < 2) return 0;

            // Catch the MTP stream up on committed positions: position i
            // consumes (token[i+1], trunk h[i]); the frontier token
            // (context's last element, not yet forwarded) seeds the draft
            // chain below instead.
            while (self.mtp_fed + 2 < context.len and self.mtp_fed + 1 < cached) : (self.mtp_fed += 1) {
                var logits = try self.model.mtpDraftStep(self.ctx, &self.mtp_cache, context[self.mtp_fed + 1], self.hiddens.items[self.mtp_fed * hidden ..][0..hidden], self.h_scratch);
                logits.deinit();
            }

            // Draft chain from the frontier: (frontier token, h[cached-1]),
            // then the MTP layer's own hidden recurrence. Rewind the
            // speculative MTP positions afterwards.
            defer self.mtp_cache.truncate(self.mtp_fed);
            const n = @min(self.depth, buf.len);
            @memcpy(self.h_prev, self.hiddens.items[(cached - 1) * hidden ..][0..hidden]);
            var prev = context[context.len - 1];
            for (buf[0..n]) |*slot| {
                var logits = try self.model.mtpDraftStep(self.ctx, &self.mtp_cache, prev, self.h_prev, self.h_scratch);
                defer logits.deinit();
                slot.* = argmaxRow(try logits.dataConst());
                prev = slot.*;
                @memcpy(self.h_prev, self.h_scratch);
            }
            return n;
        }

        /// Newly committed tokens: their trunk hiddens are the leading
        /// rows of the forward that verified them (the decoder's truncate
        /// keeps exactly `committed.len` of its rows).
        fn vtObserve(ptr: *anyopaque, committed: []const usize) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            const hidden = self.model.config.hidden_size;
            self.hiddens.appendSlice(self.allocator, self.model.step_hiddens[0 .. committed.len * hidden]) catch |err| {
                self.err = err;
            };
        }

        fn argmaxRow(row: []const f32) usize {
            var best: usize = 0;
            var best_v: f32 = -std.math.inf(f32);
            for (row, 0..) |v, i| {
                if (v > best_v) {
                    best_v = v;
                    best = i;
                }
            }
            return best;
        }
    };
}
