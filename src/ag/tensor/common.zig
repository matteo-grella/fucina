//! Accessors shared by every `Tensor(...)` branch: lifetime, raw access,
//! scalar read-out, and the tag/shape queries. Written once over
//! `Self.dtype`; the gradient-carrying branches (f32, f16, bf16) are
//! recognized by their `grad_state` field, so `deinit` honors exec-scope
//! borrows and `data` refuses mutable access to a tensor that requires
//! gradients. A mixin over the tensor struct; aliased back onto it in
//! ../tensor.zig.

const tensor_mod = @import("../../tensor.zig");
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
        const Elem = RawT.Element;
        const has_grad = @hasField(Self, "grad_state");

        pub fn deinit(self: *Self) void {
            if (comptime has_grad) {
                if (self.scope_owned) return; // borrow: the exec scope owns value + node
            }
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
            return (try self.value.dataConstChecked())[0];
        }

        /// Mutable element view of a contiguous tensor. Refused on a tensor
        /// that requires gradients (`error.MutableDataRequiresNoGrad`): the
        /// autograd tape assumes values are not mutated behind it.
        pub fn data(self: *Self) ![]Elem {
            if (comptime has_grad) {
                if (self.requiresGrad()) return error.MutableDataRequiresNoGrad;
            }
            return self.value.dataChecked();
        }

        pub fn dataConst(self: *const Self) ![]const Elem {
            return self.value.dataConstChecked();
        }

        pub fn copyTo(self: *const Self, dst: []Elem) !void {
            return self.value.copyTo(dst);
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
