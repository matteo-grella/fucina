//! The LLM trainers' resume state, persisted through the generic
//! training-checkpoint frame (`fucina.training_checkpoint`): the
//! version/step/seed header plus optional LoRA, gradient-accumulation,
//! dataloader, and evolution-strategies fields. The frame is generic over
//! the state struct; this file is the concrete struct the LoRA and ES
//! trainer CLIs share. Public as `models.train.trainer_state`.

const std = @import("std");
const fucina = @import("fucina");

const training_checkpoint = fucina.training_checkpoint;

pub const TrainerState = struct {
    version: u32 = 1,
    step: u64 = 0,
    seed: u64 = 0,
    lora_rank: ?u64 = null,
    lora_alpha: ?f64 = null,
    lora_dropout_p: ?f64 = null,
    learning_rate: ?f64 = null,
    /// Gradient-accumulation window size the loop trained with (optional,
    /// like `lora_rank`). Checkpoints must be written at window boundaries —
    /// accumulated gradients are never serialized — so on resume `step`
    /// (micro-batch count) satisfies `step % accum_steps == 0`.
    accum_steps: ?u64 = null,
    /// Dataloader stream position (`models.text.data.Loader.State`), optional as a
    /// triple: the epoch permutation is a pure function of
    /// (data_seed, data_epoch), so these three fields fully reconstruct the
    /// sample order on resume.
    data_seed: ?u64 = null,
    data_epoch: ?u64 = null,
    data_index: ?u64 = null,
    /// Evolution-strategies trainer state (`fucina.es`), optional like
    /// `lora_rank`: sigma/alpha/population pin the run configuration
    /// (validate them on resume), `es_noise` pins the noise scheme (STABLE
    /// on-disk mapping — 0 = iid, 1 = correlated; never `@intFromEnum`), and
    /// `es_iteration` restores the member-seed stream position — (seed,
    /// iteration, population, scheme) fully regenerate the population, so
    /// nothing else needs serializing. Flags that do NOT affect the noise
    /// contract (restore mode, reward) are re-passed on the CLI like
    /// `--shuffle`.
    es_sigma: ?f64 = null,
    es_alpha: ?f64 = null,
    es_population: ?u64 = null,
    es_noise: ?u64 = null,
    /// 1 = mirrored (antithetic) pairs, 0/absent = independent members.
    /// Part of the noise contract like `es_noise`.
    es_antithetic: ?u64 = null,
    /// Anchored weight decay: 0/absent = none, 1 = l1, 2 = l2 (stable
    /// mapping), with `es_anchor_lambda` the configured lambda. The anchor
    /// ITSELF is not serialized — it is reconstructable (reload the
    /// pretrained weights / re-init adapters from the seed) and must be
    /// re-captured BEFORE loading the checkpointed parameters on resume.
    es_anchor_decay: ?u64 = null,
    es_anchor_lambda: ?f64 = null,
    es_iteration: ?u64 = null,
};

test "trainer state serializes with the pinned json shape" {
    // Byte-format pin: the writer/parser bodies are comptime-generated, so
    // this is what keeps them from co-drifting to a DIFFERENT format that
    // still roundtrips (key spelling, emission order, layout).
    var aw = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer aw.deinit();
    try training_checkpoint.writeTrainerStateJson(TrainerState{ .step = 12, .seed = 34, .lora_rank = 8, .learning_rate = 1e-3, .es_iteration = 17 }, &aw.writer);
    try std.testing.expectEqualStrings(
        "{\n" ++
            "  \"format\": \"fucina.training_checkpoint\",\n" ++
            "  \"version\": 1,\n" ++
            "  \"step\": 12,\n" ++
            "  \"seed\": 34,\n" ++
            "  \"lora_rank\": 8,\n" ++
            "  \"learning_rate\": 0.001,\n" ++
            "  \"es_iteration\": 17\n" ++
            "}\n",
        aw.written(),
    );
}

test "trainer state roundtrips through directory sentinel" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var path_buf: [128]u8 = undefined;
    const dir_path = try std.fmt.bufPrint(&path_buf, "training_checkpoint_test_{d}", .{std.Io.Clock.real.now(io).nanoseconds});
    defer std.Io.Dir.cwd().deleteTree(io, dir_path) catch {};

    try training_checkpoint.beginSave(allocator, io, dir_path);
    try training_checkpoint.saveTrainerState(allocator, io, dir_path, TrainerState{
        .step = 12,
        .seed = 34,
        .lora_rank = 8,
        .lora_alpha = 16,
        .lora_dropout_p = 0.125,
        .learning_rate = 1e-3,
        .accum_steps = 4,
        .data_seed = 42,
        .data_epoch = 2,
        .data_index = 3,
        .es_sigma = 0.001,
        .es_alpha = 0.0005,
        .es_population = 30,
        .es_noise = 1,
        .es_antithetic = 1,
        .es_anchor_decay = 2,
        .es_anchor_lambda = 10.0,
        .es_iteration = 17,
    });
    const loaded = try training_checkpoint.loadTrainerState(TrainerState, allocator, io, dir_path);
    try std.testing.expectEqual(@as(u32, 1), loaded.version);
    try std.testing.expectEqual(@as(u64, 12), loaded.step);
    try std.testing.expectEqual(@as(u64, 34), loaded.seed);
    try std.testing.expectEqual(@as(?u64, 8), loaded.lora_rank);
    try std.testing.expectEqual(@as(?f64, 16), loaded.lora_alpha);
    try std.testing.expectEqual(@as(?f64, 0.125), loaded.lora_dropout_p);
    try std.testing.expectEqual(@as(?f64, 1e-3), loaded.learning_rate);
    try std.testing.expectEqual(@as(?u64, 4), loaded.accum_steps);
    try std.testing.expectEqual(@as(?u64, 42), loaded.data_seed);
    try std.testing.expectEqual(@as(?u64, 2), loaded.data_epoch);
    try std.testing.expectEqual(@as(?u64, 3), loaded.data_index);
    try std.testing.expectEqual(@as(?f64, 0.001), loaded.es_sigma);
    try std.testing.expectEqual(@as(?f64, 0.0005), loaded.es_alpha);
    try std.testing.expectEqual(@as(?u64, 30), loaded.es_population);
    try std.testing.expectEqual(@as(?u64, 1), loaded.es_noise);
    try std.testing.expectEqual(@as(?u64, 1), loaded.es_antithetic);
    try std.testing.expectEqual(@as(?u64, 2), loaded.es_anchor_decay);
    try std.testing.expectEqual(@as(?f64, 10.0), loaded.es_anchor_lambda);
    try std.testing.expectEqual(@as(?u64, 17), loaded.es_iteration);

    // Absent optionals stay null (older checkpoints without accum_steps or
    // dataloader state).
    try training_checkpoint.beginSave(allocator, io, dir_path);
    try training_checkpoint.saveTrainerState(allocator, io, dir_path, TrainerState{ .step = 3, .seed = 7 });
    const bare = try training_checkpoint.loadTrainerState(TrainerState, allocator, io, dir_path);
    try std.testing.expectEqual(@as(?u64, null), bare.accum_steps);
    try std.testing.expectEqual(@as(?u64, null), bare.lora_rank);
    try std.testing.expectEqual(@as(?u64, null), bare.data_seed);
    try std.testing.expectEqual(@as(?u64, null), bare.data_epoch);
    try std.testing.expectEqual(@as(?u64, null), bare.data_index);
    try std.testing.expectEqual(@as(?f64, null), bare.es_sigma);
    try std.testing.expectEqual(@as(?u64, null), bare.es_iteration);
}
