//! Accessors shared by every `Tensor(...)` branch: lifetime, raw access,
//! scalar read-out, and the tag/shape queries. Written once over
//! `Self.dtype`; every branch carries the exec-scope borrow flag, and the
//! gradient-carrying branches (f32, f16, bf16) are recognized by their
//! non-void `grad_state` field (on f64 the field is `void` — no slot), so
//! `deinit` releases the graph reference and `data` refuses mutable access
//! to a tensor that requires gradients. A mixin over the tensor struct;
//! aliased back onto it in ../tensor.zig.

const tensor_mod = @import("../../tensor.zig");
const dtype_mod = @import("../../dtype.zig");
const tags_mod = @import("../../tags.zig");

const TensorError = tensor_mod.TensorError;
const Tag = tags_mod.Tag;
const tagIndex = tags_mod.tagIndex;
const tagIndexOrCompileError = tags_mod.tagIndexOrCompileError;

pub fn Ops(comptime Self: type) type {
    return struct {
        const dtype = Self.dtype;
        const tags = Self.axis_tags;
        const tensor_rank = Self.tensor_rank;
        const RawT = tensor_mod.TensorOf(dtype);
        /// What the accessors speak: the VALUE type for the bit-storage
        /// floats (`Bf16`, `F8E4M3`, `F8E5M2` — layout-identical to their
        /// bits), the raw element for every other dtype. The raw tensor
        /// layer underneath stays on bits; this boundary converts by
        /// reinterpretation only.
        const Elem = dtype_mod.Element(dtype);
        const has_grad = @hasField(Self, "grad_state") and @FieldType(Self, "grad_state") != void;

        fn toElem(raw: RawT.Element) Elem {
            if (comptime Elem == RawT.Element) return raw;
            return @bitCast(raw);
        }

        /// Release the handle: the value buffer's reference and, on the
        /// gradient-carrying branches, the graph node's. A scope-owned
        /// handle is a borrow and this is a no-op for both (the scope
        /// releases at close).
        pub fn deinit(self: *Self) void {
            if (self.scope_owned) return;
            self.value.deinit();
            if (comptime has_grad) {
                if (self.grad_state) |state| state.release();
            }
            self.* = undefined;
        }

        pub fn asRawTensor(self: *const Self) *const RawT {
            return &self.value;
        }

        pub fn item(self: *const Self) !Elem {
            if (!self.value.isScalar()) return TensorError.InvalidShape;
            return toElem((try self.value.dataConstChecked())[0]);
        }

        /// Mutable element view of a contiguous tensor. Refused on a tensor
        /// that requires gradients (`error.MutableDataRequiresNoGrad`): the
        /// autograd tape assumes values are not mutated behind it.
        pub fn data(self: *Self) ![]Elem {
            if (comptime has_grad) {
                if (self.requiresGrad()) return error.MutableDataRequiresNoGrad;
            }
            return @ptrCast(try self.value.dataChecked());
        }

        pub fn dataConst(self: *const Self) ![]const Elem {
            return @ptrCast(try self.value.dataConstChecked());
        }

        pub fn copyTo(self: *const Self, dst: []Elem) !void {
            return self.value.copyTo(@ptrCast(dst));
        }

        pub fn requiresGrad(self: *const Self) bool {
            if (comptime has_grad) return self.grad_state != null;
            return false;
        }

        pub fn axis(comptime tag: Tag) usize {
            return tagIndexOrCompileError(tags, tag);
        }

        pub fn hasTag(comptime tag: Tag) bool {
            return comptime tagIndex(tags, tag) != null;
        }

        pub fn dim(self: *const Self, comptime tag: Tag) usize {
            return self.value.shape.at(axis(tag));
        }

        pub fn shape(self: *const Self) [tensor_rank]usize {
            var out: [tensor_rank]usize = undefined;
            inline for (0..tensor_rank) |i| {
                out[i] = self.value.shape.at(i);
            }
            return out;
        }

        /// True when the storage is dense row-major in logical order
        /// (innermost stride 1), the layout `data`/`dataConst` require;
        /// false for strided views (permutes, broadcasts, inner narrows).
        pub fn isContiguous(self: *const Self) bool {
            return self.value.isContiguous();
        }
    };
}
