//! Behavioral tests for cartridge fleets (`cartridge_fleet.zig`): manifest
//! JSON roundtrip, the cosine chunk index (selection order, document
//! dedupe, safetensors roundtrip), the rotation policy's ordering, and the
//! budget manager's bit-consistency guarantee — an evict/reload cycle
//! (rows + Adam moments through disk) must continue training EXACTLY as if
//! the cartridge had never left memory.

const std = @import("std");
const fucina = @import("fucina");
const fleet_mod = @import("cartridge_fleet.zig");
const cartridge = @import("cartridge.zig");
const qwen3_train = @import("qwen3/train.zig");
const scaffolding = @import("qwen3/train_tests.zig");

const ExecContext = fucina.ExecContext;

test "manifest roundtrips through JSON (names escaped)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var manifest = fleet_mod.Manifest.init(allocator, 128);
    defer manifest.deinit();
    manifest.embed_chunk = 64;
    manifest.embed_dim = 32;
    manifest.rounds = 7;
    try std.testing.expectEqual(@as(usize, 0), try manifest.addDoc("docs/README.md", 5000));
    try std.testing.expectEqual(@as(usize, 1), try manifest.addDoc("a \"quoted\"\\name\n", 1234));
    manifest.docs.items[1].steps = 42;

    var buf: [8 * 1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try manifest.write(&writer);

    var back = try fleet_mod.Manifest.parse(allocator, writer.buffered());
    defer back.deinit();
    try std.testing.expectEqual(manifest.p, back.p);
    try std.testing.expectEqual(manifest.frozen_prefix, back.frozen_prefix);
    try std.testing.expectEqual(manifest.embed_chunk, back.embed_chunk);
    try std.testing.expectEqual(manifest.embed_dim, back.embed_dim);
    try std.testing.expectEqual(manifest.rounds, back.rounds);
    try std.testing.expectEqual(manifest.docs.items.len, back.docs.items.len);
    for (manifest.docs.items, back.docs.items) |*want, *got| {
        try std.testing.expectEqualStrings(want.name, got.name);
        try std.testing.expectEqualStrings(want.cart_file, got.cart_file);
        try std.testing.expectEqualStrings(want.opt_file, got.opt_file);
        try std.testing.expectEqual(want.tokens, got.tokens);
        try std.testing.expectEqual(want.steps, got.steps);
    }
    try std.testing.expectEqual(@as(?usize, 1), back.findDoc("a \"quoted\"\\name\n"));
    try std.testing.expectEqual(@as(?usize, null), back.findDoc("missing"));
}

test "embed index selects documents by best-chunk cosine (dedup, cap, ties)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var index = fleet_mod.EmbedIndex.init(allocator, 2);
    defer index.deinit();
    // A centroid-free arrangement (the four unit rows sum to zero), so the
    // geometry is easy to read: doc 0 owns +x and -y, doc 1 owns +y, doc 2
    // owns -x. Magnitudes are irrelevant (rows normalize on append).
    try index.append(0, &.{ 3, 0 });
    try index.append(0, &.{ 0, -1 });
    try index.append(1, &.{ 0, 1 });
    try index.append(2, &.{ -1, 0 });

    // Selection requires a finalized index.
    try std.testing.expectError(fleet_mod.Error.InvalidIndex, index.topDocs(allocator, &.{ 1, 0 }, 2, 2));
    try index.finalize();
    try std.testing.expectError(fleet_mod.Error.InvalidIndex, index.finalize());
    try std.testing.expectError(fleet_mod.Error.InvalidIndex, index.append(0, &.{ 1, 0 }));

    // Query along +x: doc 0 first (best chunk = exact match), then doc 1
    // (orthogonal, score 0 — tied with doc 0's second chunk, and the tie
    // keeps the lower chunk index through the dedupe), then doc 2.
    const hits = try index.topDocs(allocator, &.{ 2, 0 }, 4, 8);
    defer allocator.free(hits);
    try std.testing.expectEqual(@as(usize, 3), hits.len);
    try std.testing.expectEqual(@as(usize, 0), hits[0].doc);
    try std.testing.expectApproxEqAbs(@as(f32, 1), hits[0].score, 1e-6);
    try std.testing.expectEqual(@as(usize, 1), hits[1].doc);
    try std.testing.expectEqual(@as(usize, 2), hits[2].doc);

    // max_docs caps the dedup pass; k_chunks caps the chunk scan.
    const capped = try index.topDocs(allocator, &.{ 2, 0 }, 4, 1);
    defer allocator.free(capped);
    try std.testing.expectEqual(@as(usize, 1), capped.len);
    try std.testing.expectEqual(@as(usize, 0), capped[0].doc);

    const one_chunk = try index.topDocs(allocator, &.{ 1, 0 }, 1, 8);
    defer allocator.free(one_chunk);
    try std.testing.expectEqual(@as(usize, 1), one_chunk.len);
    try std.testing.expectEqual(@as(usize, 0), one_chunk[0].doc);

    // Dimension mismatches are rejected.
    try std.testing.expectError(fleet_mod.Error.InvalidIndex, index.topDocs(allocator, &.{ 1, 0, 0 }, 2, 2));

    // docScores: per-doc best-chunk cosines under the same centering as
    // topDocs (adaptive serving's hysteresis probe); a doc with no chunks
    // reports -1.
    var scores: [3]f32 = undefined;
    try index.docScores(allocator, &.{ 2, 0 }, &.{ 0, 1, 7 }, &scores);
    try std.testing.expectApproxEqAbs(@as(f32, 1), scores[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), scores[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -1), scores[2], 1e-6);
}

test "finalize centers rows and queries by the unit-row centroid" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var index = fleet_mod.EmbedIndex.init(allocator, 2);
    defer index.deinit();
    try index.append(0, &.{ 1, 0 });
    try index.append(1, &.{ 0, 1 });
    try index.finalize();

    // Centroid (0.5, 0.5); centered rows are antipodal unit vectors.
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), index.centroid[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), index.centroid[1], 1e-6);
    const s = @sqrt(2.0) / 2.0;
    try std.testing.expectApproxEqAbs(@as(f32, s), index.vecs.items[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -s), index.vecs.items[1], 1e-6);

    // A +x query centers to doc 0's row exactly: cosine 1 vs doc 1's -1.
    const hits = try index.topDocs(allocator, &.{ 1, 0 }, 2, 2);
    defer allocator.free(hits);
    try std.testing.expectEqual(@as(usize, 2), hits.len);
    try std.testing.expectEqual(@as(usize, 0), hits[0].doc);
    try std.testing.expectApproxEqAbs(@as(f32, 1), hits[0].score, 1e-6);
    try std.testing.expectEqual(@as(usize, 1), hits[1].doc);
    try std.testing.expectApproxEqAbs(@as(f32, -1), hits[1].score, 1e-6);
}

test "embed index roundtrips through safetensors bitwise" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var index = fleet_mod.EmbedIndex.init(allocator, 3);
    defer index.deinit();
    try index.append(4, &.{ 0.5, -0.25, 2 });
    try index.append(0, &.{ -1, 3, 0.125 });
    try index.finalize();

    var buf: [16 * 1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try index.serialize(allocator, &writer);

    var back = try fleet_mod.EmbedIndex.initFromBytes(allocator, writer.buffered());
    defer back.deinit();
    try std.testing.expect(back.finalized());
    try std.testing.expectEqual(index.dim, back.dim);
    try std.testing.expectEqualSlices(f32, index.vecs.items, back.vecs.items);
    try std.testing.expectEqualSlices(f32, index.centroid, back.centroid);
    try std.testing.expectEqualSlices(u32, index.chunk_doc.items, back.chunk_doc.items);
}

test "rotation policy orders evictions and loads by step counts" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    const steps = [_]u64{ 5, 9, 9, 1, 3, 0 };
    const resident = [_]bool{ true, true, true, false, false, false };

    // Evictions: most-stepped residents first, ties by lower doc id.
    const victims = try fleet_mod.pickEvictions(allocator, &steps, &resident, 2);
    defer allocator.free(victims);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, victims);

    // Loads: least-stepped absentees first.
    const arrivals = try fleet_mod.pickLoads(allocator, &steps, &resident, 2);
    defer allocator.free(arrivals);
    try std.testing.expectEqualSlices(usize, &.{ 5, 3 }, arrivals);

    // Requesting more than available returns what exists.
    const all_absent = try fleet_mod.pickLoads(allocator, &steps, &resident, 10);
    defer allocator.free(all_absent);
    try std.testing.expectEqualSlices(usize, &.{ 5, 3, 4 }, all_absent);

    const none = try fleet_mod.pickEvictions(allocator, &steps, &resident, 0);
    defer allocator.free(none);
    try std.testing.expectEqual(@as(usize, 0), none.len);
}

const CartridgeTrainer = qwen3_train.Trainer(.{ .q = false, .v = false });
const no_lora = fucina.lora.Config{ .rank = 1, .alpha = 1 };

const doc_tokens = [_]usize{ 14, 6, 25, 0, 13, 1, 26, 22, 5, 29 };
const train_seq = doc_tokens[4..];
const fleet_p = 4;

fn uniqueDir(buf: []u8, tag: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "cartridge_fleet_test_{s}_{d}", .{ tag, std.Io.Clock.real.now(std.testing.io).nanoseconds });
}

/// One distillation step through the fleet's resident optimizer, using the
/// warm-up lr exactly like the CLI loop does.
fn distillStep(
    ctx: *ExecContext,
    trainer: *CartridgeTrainer,
    fleet: *fleet_mod.Fleet,
    resident_idx: usize,
    targets: cartridge.DistillTargets,
) !void {
    const resident = &fleet.residents.items[resident_idx];
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        var loss = try trainer.distillLossExt(ctx, train_seq, .{ .cartridges = &.{&resident.cart} }, targets, .{});
        defer loss.deinit();
        try loss.backward(ctx);
    }
    resident.opt.config.lr = fleet.residentLr(resident_idx);
    try resident.opt.step(ctx);
    resident.opt.zeroGrad();
    fleet.noteStep(resident.doc);
}

test "evict/reload continues training bit-identically (rows + Adam moments)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    const io = std.testing.io;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 3);
    defer model.deinit();
    var trainer = try CartridgeTrainer.init(&ctx, &model, no_lora, 7);
    defer trainer.deinit();

    // Teacher targets shared by both arms.
    var teacher = try trainer.evalLogits(&ctx, &doc_tokens);
    defer teacher.deinit();
    const vocab = model.config.vocab_size;
    const teacher_data = try teacher.dataConst();
    var builder = cartridge.TargetsBuilder.init(allocator);
    defer builder.deinit();
    for (1..train_seq.len) |j| {
        const row = teacher_data[(fleet_p + j - 1) * vocab ..][0..vocab];
        try builder.appendRow(j, row, 5, 0.99);
    }

    var dir_buf: [128]u8 = undefined;
    const dir = try uniqueDir(&dir_buf, "bitid");
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const policy = fleet_mod.RotationPolicy{ .budget = 1, .every = 0, .warmup = 3 };

    // Arm 1 (interrupted): 2 steps, evict to disk, reload, 2 more steps.
    var interrupted: []f32 = undefined;
    {
        var manifest = fleet_mod.Manifest.init(allocator, fleet_p);
        errdefer manifest.deinit();
        _ = try manifest.addDoc("doc-a", doc_tokens.len);
        var fleet = try fleet_mod.Fleet.create(allocator, io, dir, manifest, 2e-2, policy);
        defer fleet.deinit();

        const cart = try trainer.initCartridge(&ctx, doc_tokens[0..fleet_p], 1);
        var idx = try fleet.adoptResident(0, cart);
        try distillStep(&ctx, &trainer, &fleet, idx, builder.targets());
        try distillStep(&ctx, &trainer, &fleet, idx, builder.targets());

        try fleet.evictResident(io, idx);
        try std.testing.expectEqual(@as(usize, 0), fleet.residents.items.len);
        idx = try fleet.loadResident(&ctx, io, 0);
        // The reloaded resident resumes the warm-up ramp mid-flight
        // (entered_step moved), so pin the lr to the uninterrupted
        // trajectory's value for the equality check.
        fleet.residents.items[idx].entered_step = 0;

        try distillStep(&ctx, &trainer, &fleet, idx, builder.targets());
        try distillStep(&ctx, &trainer, &fleet, idx, builder.targets());
        try std.testing.expectEqual(@as(u64, 4), fleet.manifest.docs.items[0].steps);

        interrupted = try allocator.dupe(f32, try fleet.residents.items[idx].cart.layers[0].k.dataConst());
        try fleet.saveAll(io);
    }
    defer allocator.free(interrupted);

    // Arm 2 (uninterrupted): 4 straight steps, never leaving memory.
    {
        var manifest = fleet_mod.Manifest.init(allocator, fleet_p);
        errdefer manifest.deinit();
        _ = try manifest.addDoc("doc-a", doc_tokens.len);
        var dir2_buf: [128]u8 = undefined;
        const dir2 = try uniqueDir(&dir2_buf, "straight");
        defer std.Io.Dir.cwd().deleteTree(io, dir2) catch {};
        var fleet = try fleet_mod.Fleet.create(allocator, io, dir2, manifest, 2e-2, policy);
        defer fleet.deinit();

        const cart = try trainer.initCartridge(&ctx, doc_tokens[0..fleet_p], 1);
        const idx = try fleet.adoptResident(0, cart);
        for (0..4) |_| try distillStep(&ctx, &trainer, &fleet, idx, builder.targets());

        const straight = try fleet.residents.items[idx].cart.layers[0].k.dataConst();
        try std.testing.expectEqualSlices(f32, straight, interrupted);
    }

    // Reopen the saved fleet: manifest state survived.
    {
        var fleet = try fleet_mod.Fleet.open(allocator, io, dir, 2e-2, policy);
        defer fleet.deinit();
        try std.testing.expectEqual(@as(usize, 1), fleet.manifest.docs.items.len);
        try std.testing.expectEqual(@as(u64, 4), fleet.manifest.docs.items[0].steps);
        try std.testing.expectEqualStrings("doc-a", fleet.manifest.docs.items[0].name);
    }
}

test "budget rotation swaps most-trained residents for least-trained absentees" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();
    const io = std.testing.io;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try scaffolding.buildTinyModel(&ctx, 3);
    defer model.deinit();
    var trainer = try CartridgeTrainer.init(&ctx, &model, no_lora, 7);
    defer trainer.deinit();

    var dir_buf: [128]u8 = undefined;
    const dir = try uniqueDir(&dir_buf, "rotate");
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    var manifest = fleet_mod.Manifest.init(allocator, fleet_p);
    errdefer manifest.deinit();
    _ = try manifest.addDoc("doc-a", 100);
    _ = try manifest.addDoc("doc-b", 100);
    _ = try manifest.addDoc("doc-c", 100);
    var fleet = try fleet_mod.Fleet.create(allocator, io, dir, manifest, 1e-2, .{
        .budget = 2,
        .every = 1,
        .evict_fraction = 0.5,
        .warmup = 0,
    });
    defer fleet.deinit();

    // Initialize all three docs on disk (the CLI's init pass), then keep
    // a and b resident.
    for (0..3) |doc| {
        const cart = try trainer.initCartridge(&ctx, doc_tokens[0..fleet_p], 1);
        const idx = try fleet.adoptResident(doc, cart);
        try fleet.evictResident(io, idx);
    }
    _ = try fleet.loadResident(&ctx, io, 0);
    _ = try fleet.loadResident(&ctx, io, 1);

    // Doc 0 trained more than doc 1; doc 2 never trained.
    fleet.manifest.docs.items[0].steps = 6;
    fleet.manifest.docs.items[1].steps = 2;

    fleet.manifest.rounds += 1;
    const loaded = try fleet.maybeRotate(&ctx, io, null);
    try std.testing.expectEqual(@as(usize, 1), loaded);

    // Doc 0 (most-trained) left; doc 2 (least-trained) arrived; doc 1 stayed.
    try std.testing.expectEqual(@as(?usize, null), fleet.residentIndex(0));
    try std.testing.expect(fleet.residentIndex(1) != null);
    try std.testing.expect(fleet.residentIndex(2) != null);
    try std.testing.expectEqual(@as(usize, 2), fleet.residents.items.len);

    // Warm-up ramp: a fresh entrant at warmup 4 ramps 1/4, 2/4, ... of base.
    fleet.policy.warmup = 4;
    const entrant = fleet.residentIndex(2).?;
    fleet.residents.items[entrant].entered_step = fleet.manifest.docs.items[2].steps;
    try std.testing.expectApproxEqAbs(@as(f32, 1e-2 / 4.0), fleet.residentLr(entrant), 1e-9);
    fleet.noteStep(2);
    try std.testing.expectApproxEqAbs(@as(f32, 1e-2 / 2.0), fleet.residentLr(entrant), 1e-9);

    try fleet.saveAll(io);
    try std.testing.expectEqual(@as(usize, 0), fleet.residents.items.len);
}
