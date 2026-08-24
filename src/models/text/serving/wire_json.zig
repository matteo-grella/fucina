//! Shared head of the wire-dialect request parsers (`openai.zig`,
//! `anthropic.zig`): error accumulation, typed field access over the
//! dynamic JSON `Value`, and the forced-tool_choice plumbing. Each
//! dialect's `Parser` holds one `Head` and forwards to it one line per
//! method, so the dialect files keep only what actually differs on the
//! wire (roles, content blocks, the field surfaces).

const std = @import("std");
const chat = @import("../chat.zig");
const types = @import("contract.zig");
const toolcall = @import("toolcall.zig");

const Allocator = std.mem.Allocator;
const Value = std.json.Value;

pub const Error = error{ Invalid, OutOfMemory };

/// Declarations in both forms the pipeline needs: the serialized
/// objects the hermes system block embeds, and the (name, schema)
/// pairs the forced-call grammar is built from.
pub const ToolSet = struct {
    json: []const []const u8 = &.{},
    decls: []const toolcall.Decl = &.{},
};

/// The dialect-independent parser state, generic over the dialect's error
/// shape (`openai.ErrorInfo`, which the Anthropic layer reuses; it must
/// provide `invalid(message, param)`).
pub fn Head(comptime ErrorInfo: type) type {
    return struct {
        const Self = @This();

        arena: Allocator,
        obj: std.json.ObjectMap,
        info: types.Info,
        err: ?ErrorInfo = null,
        /// Set by a forced tool_choice ("required" or a named function on
        /// the OpenAI dialects, `any`/`tool` on Anthropic): the dialect
        /// compiles these into the forced-call grammar once the other
        /// constraints are resolved.
        forced_decls: ?[]const toolcall.Decl = null,

        pub fn fail(self: *Self, info: ErrorInfo) Error {
            if (self.err == null) self.err = info;
            return error.Invalid;
        }

        pub fn failInvalid(self: *Self, message: []const u8, param: ?[]const u8) Error {
            return self.fail(ErrorInfo.invalid(message, param));
        }

        // ---- typed field access over the dynamic Value ----

        pub fn optField(self: *Self, obj: std.json.ObjectMap, name: []const u8) ?Value {
            _ = self;
            const v = obj.get(name) orelse return null;
            if (v == .null) return null;
            return v;
        }

        pub fn optString(self: *Self, obj: std.json.ObjectMap, name: []const u8) Error!?[]const u8 {
            const v = self.optField(obj, name) orelse return null;
            if (v != .string) return self.failInvalid("expected a string", name);
            return v.string;
        }

        pub fn optBool(self: *Self, obj: std.json.ObjectMap, name: []const u8) Error!?bool {
            const v = self.optField(obj, name) orelse return null;
            if (v != .bool) return self.failInvalid("expected a boolean", name);
            return v.bool;
        }

        pub fn optF32(self: *Self, obj: std.json.ObjectMap, name: []const u8, min: f32, max: f32) Error!?f32 {
            const v = self.optField(obj, name) orelse return null;
            const x: f64 = switch (v) {
                .integer => |i| @floatFromInt(i),
                .float => |f| f,
                else => return self.failInvalid("expected a number", name),
            };
            if (!std.math.isFinite(x) or x < min or x > max) return self.failInvalid("value out of range", name);
            return @floatCast(x);
        }

        pub fn optInt(self: *Self, obj: std.json.ObjectMap, name: []const u8, min: i64) Error!?i64 {
            const v = self.optField(obj, name) orelse return null;
            if (v != .integer) return self.failInvalid("expected an integer", name);
            if (v.integer < min) return self.failInvalid("value out of range", name);
            return v.integer;
        }

        // ---- tool calling (hermes backends; `toolcall.zig` renders) ----

        /// Flush accumulated tool-result sections as one user turn (Qwen3's
        /// template shape: consecutive results share the turn).
        pub fn flushToolResponses(self: *Self, messages: *std.ArrayList(chat.Message), fold: *std.ArrayList(u8)) Error!void {
            if (fold.items.len == 0) return;
            try messages.append(self.arena, .{ .role = .user, .content = try fold.toOwnedSlice(self.arena) });
        }

        /// Force the single declaration called `name`. `noun` is the
        /// dialect's word for a declaration in the error message
        /// ("function" on the OpenAI dialects, "tool" on Anthropic).
        pub fn forceNamed(self: *Self, comptime noun: []const u8, decls: []const toolcall.Decl, name: []const u8) Error!void {
            for (decls) |d| {
                if (std.mem.eql(u8, d.name, name)) {
                    const one = try self.arena.alloc(toolcall.Decl, 1);
                    one[0] = d;
                    return self.forceDecls(one);
                }
            }
            return self.failInvalid("tool_choice names an undeclared " ++ noun, "tool_choice");
        }

        pub fn forceDecls(self: *Self, decls: []const toolcall.Decl) Error!void {
            if (decls.len == 0)
                return self.failInvalid("tool_choice requires tools to be declared", "tool_choice");
            for (decls) |d| {
                if (!toolcall.plainName(d.name))
                    return self.failInvalid("tool names must use [A-Za-z0-9_.:-] characters for a forced tool_choice", "tools");
            }
            self.forced_decls = decls;
        }
    };
}
