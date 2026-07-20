var x = try Tensor(.{.d})
    .variableFromSlice(&ctx, .{3}, &.{1, 2, 3});
var c = try Tensor(.{.d})   // constant: no grad
    .fromSlice(&ctx, .{3}, &.{10, 20, 30});

// identical call site, training or inference
var y = try x.mul(&ctx, &c);
var loss = try y.sumAll(&ctx);
try loss.backward(&ctx);

var gx = (try x.grad(&ctx)).?;  // [10, 20, 30]
// a constant never accumulates
try expect((try c.grad(&ctx)) == null);
