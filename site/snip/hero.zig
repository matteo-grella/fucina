// x: [batch=1, in=2],  w: [in=2, out=1]
var x = try Tensor(.{ .batch, .in })
    .variableFromSlice(&ctx, .{1, 2}, &.{2, 3});
var w = try Tensor(.{ .in, .out })
    .variableFromSlice(&ctx, .{2, 1}, &.{4, 5});

// contract .in  =>  [batch, out]
var y = try x.dot(&ctx, &w, .in);
var loss = try y.sumAll(&ctx);

try loss.backward(&ctx);
var gx = (try x.grad(&ctx)).?;  // = wᵀ = [4, 5]
