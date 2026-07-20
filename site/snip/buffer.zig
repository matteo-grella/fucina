var first = try ctx.add(&a, &a);
const first_ptr = first.dataConst().ptr;
first.deinit();  // refcount 0 → back to the pool

// same size → the pool returns the SAME address
var second = try ctx.add(&a, &a);
try expectEqual(first_ptr,
    second.dataConst().ptr);
