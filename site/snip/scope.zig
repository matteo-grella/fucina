var w = try Tensor(.{.d})
    .variableFromSlice(&ctx, .{3}, &.{1, 2, 3});
defer w.deinit();  // params stay caller-owned

const scope = ctx.openExecScope();
defer ctx.closeExecScope(scope);
var y = try w.mul(&ctx, &w);
defer y.deinit();  // no-op: the scope owns y
var loss = try y.sumAll(&ctx);
defer loss.deinit();  // same code runs unscoped
try loss.backward(&ctx);

var gw = (try w.grad(&ctx)).?;  // [2, 4, 6]
