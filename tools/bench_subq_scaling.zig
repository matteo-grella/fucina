//! Selection-scaling probe: synthetic clustered KV at growing context
//! lengths, fixed tau, no model. Times attend() for flat vs hierarchical
//! selection to demonstrate the scaling difference: flat selection work
//! (scoring, sort, tail) grows linearly with context, the hierarchical
//! frontier walk grows with the opened set.
//!
//!   zig build bench-subq-scaling -Doptimize=ReleaseFast -- [--max-ctx N]

const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const q_heads = 16;
const kv_heads = 8;
const d = 128;
const archetypes = 64;

fn lcg(s: *u64) f32 {
    s.* = s.* *% 6364136223846793005 +% 1442695040888963407;
    const bits: u32 = @truncate(s.* >> 33);
    return @as(f32, @floatFromInt(bits)) / @as(f32, @floatFromInt(std.math.maxInt(u32))) * 2.0 - 1.0;
}

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.c_allocator;
    const io = init.io;
    var stdout_buf: [4096]u8 = undefined;
    var stdout_w = std.Io.File.stdout().writer(io, &stdout_buf);
    const stdout = &stdout_w.interface;

    const args = try init.minimal.args.toSlice(allocator);
    defer allocator.free(args);
    var max_ctx: usize = 524288;
    var tau: f32 = 0.05;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--max-ctx")) {
            i += 1;
            max_ctx = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--tau")) {
            i += 1;
            tau = try std.fmt.parseFloat(f32, args[i]);
        }
    }

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var seed: u64 = 0x2545f4914f6cdd1d;
    var arch: [archetypes * d]f32 = undefined;
    for (&arch) |*x| x.* = lcg(&seed);

    const contexts = [_]usize{ 32768, 131072, 524288 };
    for (contexts) |n| {
        if (n > max_ctx) break;
        // Clustered synthetic KV: mass concentrates so the opened set stays
        // small and selection cost dominates at scale.
        const kbuf = try allocator.alloc(f16, n * kv_heads * d);
        defer allocator.free(kbuf);
        const vbuf = try allocator.alloc(f16, n * kv_heads * d);
        defer allocator.free(vbuf);
        for (0..n) |pos| {
            const a = (pos * 2654435761) % archetypes;
            for (0..kv_heads) |h| {
                const base = (pos * kv_heads + h) * d;
                for (0..d) |j| {
                    kbuf[base + j] = @floatCast(2.0 * arch[a * d + j] + 0.1 * lcg(&seed));
                    vbuf[base + j] = @floatCast(lcg(&seed));
                }
            }
        }
        var q: [q_heads * d]f32 = undefined;
        for (0..q_heads) |h| {
            const a = (h * 7) % archetypes;
            for (0..d) |j| q[h * d + j] = 3.0 * arch[a * d + j] + 0.2 * lcg(&seed);
        }
        var out: [q_heads * d]f32 = undefined;

        inline for (.{ false, true }, 0..) |hier, mode| {
            _ = mode;
            var state = try llm.subq.State.init(allocator, 1, q_heads, kv_heads, d, .{
                .cluster_size = 128,
                .rebuild_interval = 512,
                .tau_default = 0.05,
                .packed_format = .q8_0,
                .hierarchical = hier,
            });
            defer state.deinit();
            @memset(state.taus, tau);
            const t0 = std.Io.Clock.awake.now(io).nanoseconds;
            try state.attend(&ctx, 0, &q, kbuf, vbuf, n, &out);
            const t1 = std.Io.Clock.awake.now(io).nanoseconds;
            const reps = 16;
            const opened0 = state.stat_opened_rows;
            const t2 = std.Io.Clock.awake.now(io).nanoseconds;
            for (0..reps) |_| try state.attend(&ctx, 0, &q, kbuf, vbuf, n, &out);
            const t3 = std.Io.Clock.awake.now(io).nanoseconds;
            const opened_per = @as(f64, @floatFromInt(state.stat_opened_rows - opened0)) / reps / q_heads;
            try stdout.print("ctx {d:>7} {s}: build {d:>7.2} s  attend {d:>8.3} ms  opened/head {d:>8.1} ({d:.2}% of ctx)\n", .{
                n,
                if (hier) "hier" else "flat",
                @as(f64, @floatFromInt(t1 - t0)) / 1e9,
                @as(f64, @floatFromInt(t3 - t2)) / 1e6 / reps,
                opened_per,
                opened_per * 100.0 / @as(f64, @floatFromInt(n)),
            });
            try stdout.flush();
        }
    }
}
