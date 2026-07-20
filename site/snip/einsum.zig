// q: [g, i, d] · k: [j, d] → scores: [g, i, j]
var scores = try q.einsum(&ctx, &k,
    .{ .g, .i, .j });
// roles by membership: shared & kept = batch,
// shared & dropped = contracted,
// private & dropped = summed away.

// x[s,i] · A[r,i] · B[o,r] → [s,o]: LoRA delta
var delta = try einsumMany(&ctx, .{ .s, .o },
    .{ &x, &a, &b });
