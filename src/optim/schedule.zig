//! Learning-rate schedule hook: `LrSchedule` rescales attached `config.lr`
//! fields from their captured bases, and `warmupCosineFactor` is the
//! warmup + cosine factor it is usually fed.

const std = @import("std");

const Allocator = std.mem.Allocator;

/// Attaches to `config.lr` fields (they are plain public f32s) and rescales
/// them all from their captured base values: `lr = base * factor`. Works
/// across optimizers and their fallbacks, e.g.:
///
///     var sched = optim.LrSchedule.init(allocator);
///     defer sched.deinit();
///     try sched.attach(&muon.config.lr);
///     try sched.attach(&muon.fallback.config.lr);
///     ...
///     sched.apply(optim.warmupCosineFactor(step, total_steps, warmup, 0.1));
///
/// Because the factor is a pure function of the step counter, re-applying it
/// while resuming from a checkpoint reproduces the schedule exactly.
pub const LrSchedule = struct {
    allocator: Allocator,
    entries: std.ArrayList(Entry) = .empty,

    const Entry = struct {
        lr: *f32,
        base: f64,
    };

    pub fn init(allocator: Allocator) LrSchedule {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *LrSchedule) void {
        self.entries.deinit(self.allocator);
        self.* = undefined;
    }

    /// Captures the current value as the base lr — attach before the first
    /// `apply()`, or the factor in effect gets baked into the base. The
    /// pointee must outlive the schedule. Re-attaching a pointer refreshes
    /// its base instead of duplicating the entry.
    pub fn attach(self: *LrSchedule, lr: *f32) !void {
        for (self.entries.items) |*entry| {
            if (entry.lr == lr) {
                entry.base = lr.*;
                return;
            }
        }
        try self.entries.append(self.allocator, .{ .lr = lr, .base = lr.* });
    }

    pub fn apply(self: *const LrSchedule, factor: f64) void {
        for (self.entries.items) |entry| {
            entry.lr.* = @floatCast(entry.base * factor);
        }
    }
};

/// Linear warmup from 1/warmup_steps to 1, then cosine decay to `min_factor`
/// over the remaining steps. `step` is 0-based.
pub fn warmupCosineFactor(step: u64, total_steps: u64, warmup_steps: u64, min_factor: f64) f64 {
    if (warmup_steps > 0 and step < warmup_steps) {
        return @as(f64, @floatFromInt(step + 1)) / @as(f64, @floatFromInt(warmup_steps));
    }
    if (total_steps <= warmup_steps) return min_factor;
    const num = @as(f64, @floatFromInt(step - warmup_steps));
    const den = @as(f64, @floatFromInt(total_steps - warmup_steps));
    const progress = @min(num / den, 1.0);
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + @cos(std.math.pi * progress));
}
