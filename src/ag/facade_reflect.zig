//! Facade reflection shared by the custom-VJP, checkpoint and gradcheck
//! adapters: the tuple-of-facade-pointers input contract and the f32
//! autograd-facade check, parameterized by the caller's name (for its
//! compile errors) and by the pointer shape it requires.
const std = @import("std");

/// What a caller demands of each input pointer beyond pointing at an f32
/// facade tensor.
pub const Pointer = struct {
    /// A single-item pointer (checkpoint rewraps the pointee by value).
    single_item: bool = false,
    /// A mutable pointer (gradcheck perturbs the inputs in place).
    mutable: bool = false,
};

/// The fields of an inputs tuple; `what` names the caller in the error.
pub fn inputFields(comptime Inputs: type, comptime what: []const u8) []const std.builtin.Type.StructField {
    const info = @typeInfo(Inputs);
    if (info != .@"struct" or !info.@"struct".is_tuple) {
        @compileError(what ++ " inputs must be a tuple of facade tensor pointers, got " ++ @typeName(Inputs));
    }
    return info.@"struct".fields;
}

/// The facade type behind each input pointer, in tuple order.
pub fn facadeTypes(comptime Inputs: type, comptime what: []const u8, comptime pointer: Pointer) [inputFields(Inputs, what).len]type {
    const fields = inputFields(Inputs, what);
    var types: [fields.len]type = undefined;
    for (fields, 0..) |field, i| types[i] = FacadeOf(field.type, what, pointer);
    return types;
}

/// The tuple of facade values matching an inputs tuple of pointers.
pub fn FacadeTuple(comptime Inputs: type, comptime what: []const u8, comptime pointer: Pointer) type {
    const types = facadeTypes(Inputs, what, pointer);
    return std.meta.Tuple(&types);
}

/// The facade type an input pointer points at, checked against `pointer`.
pub fn FacadeOf(comptime P: type, comptime what: []const u8, comptime pointer: Pointer) type {
    const info = @typeInfo(P);
    if (info != .pointer) @compileError(what ++ " input must be a pointer to an f32 facade tensor, got " ++ @typeName(P));
    if (pointer.single_item and info.pointer.size != .one) {
        @compileError(what ++ " input must be a single-item pointer to a facade tensor, got " ++ @typeName(P));
    }
    if (pointer.mutable and info.pointer.is_const) {
        @compileError(what ++ " input must be a mutable pointer to an f32 facade tensor, got " ++ @typeName(P));
    }
    validateFacade(info.pointer.child, what ++ " input");
    return info.pointer.child;
}

/// `T` is an f32 autograd facade tensor (a struct with `dtype == .f32`
/// and a `grad_state` field).
pub fn validateFacade(comptime T: type, comptime what: []const u8) void {
    if (@typeInfo(T) != .@"struct" or !@hasDecl(T, "dtype") or T.dtype != .f32 or !@hasField(T, "grad_state")) {
        @compileError(what ++ " must be an f32 autograd facade tensor, got " ++ @typeName(T));
    }
}
