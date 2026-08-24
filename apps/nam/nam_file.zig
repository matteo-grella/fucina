//! `.nam` model-file reader: the canonical Neural Amp Modeler format.
//!
//! A `.nam` file is one JSON document: `version`, `architecture`, `config`,
//! `weights` (flat float array), optional `metadata`, optional `sample_rate`
//! (missing ⇒ -1 "unknown"). This module ports the upstream loader contract
//! exactly — schema, legacy config spellings, version gate, and the
//! per-architecture flat-weight counts — from NeuralAmpModelerCore@e49c93e
//! (NAM/get_dsp.cpp, wavenet/model.cpp, lstm.cpp, convnet.cpp, linear.cpp)
//! cross-checked against the neural-amp-modeler@a11ed88 exporters.
//!
//! Scope: WaveNet (incl. gated/blended, grouped convs, active
//! FiLMs, nested condition_dsp, and legacy spellings), LSTM, ConvNet, Linear,
//! and SlimmableContainer (the current upstream trainer's export shape —
//! loaded at its highest-quality submodel). Slimmable WaveNet arrays fail with
//! explicit errors.

const std = @import("std");

pub const Error = error{
    InvalidJson,
    MissingField,
    InvalidField,
    UnsupportedVersion,
    UnsupportedArchitecture,
    UnsupportedFeature,
    WeightCountMismatch,
};

/// Newest file version we write and fully support (upstream
/// LATEST_FULLY_SUPPORTED_NAM_FILE_VERSION, get_dsp.h:66).
pub const latest_version = Version{ .major = 0, .minor = 7, .patch = 0 };
/// Oldest accepted version (upstream EARLIEST_SUPPORTED, get_dsp.h:67).
pub const earliest_version = Version{ .major = 0, .minor = 5, .patch = 0 };

pub const Version = struct {
    major: u32,
    minor: u32,
    patch: u32,

    pub fn parse(text: []const u8) Error!Version {
        var it = std.mem.splitScalar(u8, text, '.');
        const major = it.next() orelse return Error.InvalidField;
        const minor = it.next() orelse return Error.InvalidField;
        const patch = it.next() orelse return Error.InvalidField;
        if (it.next() != null) return Error.InvalidField;
        return .{
            .major = std.fmt.parseInt(u32, major, 10) catch return Error.InvalidField,
            .minor = std.fmt.parseInt(u32, minor, 10) catch return Error.InvalidField,
            .patch = std.fmt.parseInt(u32, patch, 10) catch return Error.InvalidField,
        };
    }

    fn lessThan(a: Version, b: Version) bool {
        if (a.major != b.major) return a.major < b.major;
        if (a.minor != b.minor) return a.minor < b.minor;
        return a.patch < b.patch;
    }
};

pub const Arch = enum { wavenet, lstm, convnet, linear };

pub const Activation = struct {
    kind: Kind,
    /// LeakyReLU / PReLU single slope.
    negative_slope: f32 = 0.01,
    /// PReLU per-channel slopes; empty ⇒ use `negative_slope` for all.
    negative_slopes: []const f32 = &.{},
    // LeakyHardtanh parameters (upstream defaults, activations.cpp:118-124).
    min_val: f32 = -1,
    max_val: f32 = 1,
    min_slope: f32 = 0.01,
    max_slope: f32 = 0.01,

    pub const Kind = enum { tanh, hardtanh, fasttanh, relu, leaky_relu, prelu, sigmoid, silu, hardswish, leaky_hardtanh, softsign };

    pub const tanh_default = Activation{ .kind = .tanh };
    pub const sigmoid_default = Activation{ .kind = .sigmoid };
};

pub const GatingMode = enum { none, gated, blended };

pub const FiLMParams = struct {
    active: bool = false,
    shift: bool = false,
    groups: usize = 1,
};

pub const WaveNetLayerArray = struct {
    input_size: usize,
    condition_size: usize,
    channels: usize,
    bottleneck: usize,
    head_out: usize,
    head_kernel: usize,
    head_bias: bool,
    dilations: []const usize,
    /// One kernel size per layer (expanded from the legacy scalar form).
    kernel_sizes: []const usize,
    /// One activation per layer (expanded from the scalar form).
    activations: []const Activation,
    gating_modes: []const GatingMode,
    /// Secondary (gate) activation per layer; meaningful only when gated.
    secondary_activations: []const Activation,
    /// The residual 1x1 (B -> C, bias). Inactive requires bottleneck == channels.
    layer1x1_active: bool,
    layer1x1_groups: usize,
    /// Per-layer head contribution conv (B -> head1x1_out, bias).
    head1x1_active: bool,
    head1x1_out: usize,
    head1x1_groups: usize,
    groups_input: usize,
    groups_input_mixin: usize,
    conv_pre_film: FiLMParams = .{},
    conv_post_film: FiLMParams = .{},
    input_mixin_pre_film: FiLMParams = .{},
    input_mixin_post_film: FiLMParams = .{},
    activation_pre_film: FiLMParams = .{},
    activation_post_film: FiLMParams = .{},
    layer1x1_post_film: FiLMParams = .{},
    head1x1_post_film: FiLMParams = .{},

    pub fn layerCount(self: *const WaveNetLayerArray) usize {
        return self.dilations.len;
    }

    /// Conv/mixin output width of layer l: 2*bottleneck when gated.
    pub fn gateWidth(self: *const WaveNetLayerArray, layer: usize) usize {
        return switch (self.gating_modes[layer]) {
            .none => self.bottleneck,
            .gated, .blended => saturatingMul(self.bottleneck, 2),
        };
    }

    /// Receptive field of the array: 1 + sum(d*(k-1)) + (head_kernel - 1).
    pub fn receptiveField(self: *const WaveNetLayerArray) usize {
        var rf: usize = 1;
        for (self.dilations, self.kernel_sizes) |d, k| {
            rf = saturatingAdd(rf, saturatingMul(d, k - 1));
        }
        return saturatingAdd(rf, self.head_kernel - 1);
    }
};

pub const WaveNetPostHead = struct {
    channels: usize,
    out_channels: usize,
    kernel_sizes: []const usize,
    activation: Activation,
};

pub const WaveNetConfig = struct {
    layers: []const WaveNetLayerArray,
    head: ?WaveNetPostHead,
    head_scale: f32,
    in_channels: usize,
    condition_dsp: ?*const ConditionDsp,

    pub fn receptiveField(self: *const WaveNetConfig) usize {
        var rf: usize = if (self.condition_dsp) |dsp| dsp.receptiveField() else 1;
        for (self.layers) |*array| rf = saturatingAdd(rf, array.receptiveField() - 1);
        if (self.head) |*head| {
            for (head.kernel_sizes) |k| rf = saturatingAdd(rf, k - 1);
        }
        return rf;
    }
};

pub const LstmConfig = struct {
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    in_channels: usize,
    out_channels: usize,
};

pub const ConvNetConfig = struct {
    channels: usize,
    dilations: []const usize,
    batchnorm: bool,
    activation: Activation,
    in_channels: usize,
    out_channels: usize,

    pub const kernel_size = 2; // hard-coded upstream (convnet.cpp:57)

    pub fn receptiveField(self: *const ConvNetConfig) usize {
        var rf: usize = 1;
        for (self.dilations) |d| rf = saturatingAdd(rf, saturatingMul(d, kernel_size - 1));
        return rf;
    }
};

pub const LinearConfig = struct {
    receptive_field: usize,
    bias: bool,
    in_channels: usize,
    out_channels: usize,
};

pub const Config = union(Arch) {
    wavenet: WaveNetConfig,
    lstm: LstmConfig,
    convnet: ConvNetConfig,
    linear: LinearConfig,
};

pub const ConditionDsp = struct {
    architecture: Arch,
    config: Config,
    weights: []const f32,
    sample_rate: f64,

    pub fn receptiveField(self: *const ConditionDsp) usize {
        return switch (self.config) {
            .wavenet => |*c| c.receptiveField(),
            .convnet => |*c| c.receptiveField(),
            .linear => |*c| c.receptive_field,
            .lstm => if (self.sample_rate > 0 and std.math.isFinite(self.sample_rate)) blk: {
                const samples = 0.5 * self.sample_rate;
                if (samples >= @as(f64, @floatFromInt(std.math.maxInt(usize)))) break :blk std.math.maxInt(usize);
                break :blk @as(usize, @intFromFloat(samples));
            } else 1,
        };
    }
};

/// Player-relevant metadata (the three keys the upstream C++ core reads,
/// get_dsp.cpp:250-260; everything else stays in `document` as an open bag).
pub const Metadata = struct {
    loudness: ?f64 = null,
    input_level_dbu: ?f64 = null,
    output_level_dbu: ?f64 = null,
    /// User-tagged gear category (open enum: amp/preamp/pedal/pedal_amp/
    /// amp_cab/amp_pedal_cab/studio, plus vendor strings). Not read by the
    /// upstream C++ core; the player uses it only for cab-advisory hints.
    /// Borrows the JSON arena string (valid for the model lifetime).
    gear_type: ?[]const u8 = null,
};

pub const SubmodelInfo = struct {
    index: usize,
    count: usize,
    max_value: f64,
};

pub const NamModel = struct {
    parsed: std.json.Parsed(std.json.Value),
    /// Original file bytes, retained for byte-faithful re-export.
    raw_bytes: []u8,
    raw_allocator: std.mem.Allocator,

    version: Version,
    /// True when only the patch component exceeds 0.7.0 — upstream loads
    /// these with a warning ("partial support").
    partial_support: bool,
    architecture: Arch,
    config: Config,
    weights: []const f32,
    /// -1 when the file does not declare a rate (old files).
    sample_rate: f64,
    metadata: Metadata,
    /// Set when the file was a SlimmableContainer: which submodel is loaded.
    submodel_info: ?SubmodelInfo,

    pub fn deinit(self: *NamModel) void {
        self.raw_allocator.free(self.raw_bytes);
        self.parsed.deinit();
        self.* = undefined;
    }

    /// The whole parsed JSON document (for metadata inspection/round-trip).
    pub fn document(self: *const NamModel) std.json.Value {
        return self.parsed.value;
    }
};

pub fn loadFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !NamModel {
    const bytes = try readWholeFile(io, allocator, path);
    // loadOwnedBytes owns `bytes` from here, including on error.
    return loadOwnedBytes(allocator, bytes);
}

fn readWholeFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file) return error.NotAFile;
    const bytes = try allocator.alloc(u8, @intCast(stat.size));
    errdefer allocator.free(bytes);
    var read_len: usize = 0;
    while (read_len < bytes.len) {
        const n = try file.readStreaming(io, &.{bytes[read_len..]});
        if (n == 0) return error.EndOfStream;
        read_len += n;
    }
    return bytes;
}

pub fn loadFromSlice(allocator: std.mem.Allocator, bytes: []const u8) !NamModel {
    const copy = try allocator.dupe(u8, bytes);
    // loadOwnedBytes owns `copy` from here, including on error.
    return loadOwnedBytes(allocator, copy);
}

/// Takes ownership of `bytes` (even on error).
fn loadOwnedBytes(allocator: std.mem.Allocator, bytes: []u8) !NamModel {
    errdefer allocator.free(bytes);

    var parsed = std.json.parseFromSlice(std.json.Value, allocator, bytes, .{}) catch return Error.InvalidJson;
    errdefer parsed.deinit();
    const arena = parsed.arena.allocator();

    const root = switch (parsed.value) {
        .object => |o| o,
        else => return Error.InvalidJson,
    };

    // Version gate (upstream CoreVersionSupportChecker, get_dsp.cpp:19-40):
    // reject < 0.5.0, reject minor > 7 or major > 0, warn-and-load when only
    // the patch exceeds 0.7.0.
    const version_value = root.get("version") orelse return Error.MissingField;
    const version_text = switch (version_value) {
        .string => |s| s,
        else => return Error.InvalidField,
    };
    const version = try Version.parse(version_text);
    if (version.lessThan(earliest_version)) return Error.UnsupportedVersion;
    if (version.major > latest_version.major or version.minor > latest_version.minor) return Error.UnsupportedVersion;
    const partial_support = latest_version.lessThan(version);

    const arch_value = root.get("architecture") orelse return Error.MissingField;
    const arch_name = switch (arch_value) {
        .string => |s| s,
        else => return Error.InvalidField,
    };

    const config_value = root.get("config") orelse return Error.MissingField;
    const config_object = switch (config_value) {
        .object => |o| o,
        else => return Error.InvalidField,
    };

    const container_rate = try optionalSampleRate(root);

    // SlimmableContainer (container.cpp): submodels = ascending-max_value
    // list of full nested .nam documents; the runtime quality knob picks
    // one. We load the HIGHEST-quality submodel (max_value 1.0, the last —
    // what upstream players run at the default knob, and the one whose
    // loudness/gain the container metadata mirrors). The current upstream
    // trainer exports every profile in this shape.
    var submodel_info: ?SubmodelInfo = null;
    var leaf_root = root;
    var leaf_arch_name = arch_name;
    var leaf_config = config_object;
    if (std.mem.eql(u8, arch_name, "SlimmableContainer")) {
        const submodels_value = config_object.get("submodels") orelse return Error.MissingField;
        const submodels = switch (submodels_value) {
            .array => |a| a,
            else => return Error.InvalidField,
        };
        if (submodels.items.len == 0) return Error.InvalidField;

        // max_values strictly ascending, last >= 1.0 (container.cpp:27-33).
        var previous = -std.math.inf(f64);
        var last_max: f64 = 0;
        for (submodels.items) |item| {
            const entry = switch (item) {
                .object => |o| o,
                else => return Error.InvalidField,
            };
            last_max = try requireNumber(entry, "max_value");
            if (last_max <= previous) return Error.InvalidField;
            previous = last_max;
        }
        if (last_max < 1.0) return Error.InvalidField;

        const chosen = submodels.items[submodels.items.len - 1].object;
        const model_value = chosen.get("model") orelse return Error.MissingField;
        leaf_root = switch (model_value) {
            .object => |o| o,
            else => return Error.InvalidField,
        };

        // The nested document carries its own version/architecture (the C++
        // loads it through a recursive get_dsp, re-gating both).
        const nested_version_value = leaf_root.get("version") orelse return Error.MissingField;
        const nested_version = try Version.parse(switch (nested_version_value) {
            .string => |s| s,
            else => return Error.InvalidField,
        });
        if (nested_version.lessThan(earliest_version)) return Error.UnsupportedVersion;
        if (nested_version.major > latest_version.major or nested_version.minor > latest_version.minor) return Error.UnsupportedVersion;

        leaf_arch_name = switch (leaf_root.get("architecture") orelse return Error.MissingField) {
            .string => |s| s,
            else => return Error.InvalidField,
        };
        leaf_config = switch (leaf_root.get("config") orelse return Error.MissingField) {
            .object => |o| o,
            else => return Error.InvalidField,
        };
        submodel_info = .{
            .index = submodels.items.len - 1,
            .count = submodels.items.len,
            .max_value = last_max,
        };
    }

    const arch: Arch = if (std.mem.eql(u8, leaf_arch_name, "WaveNet"))
        .wavenet
    else if (std.mem.eql(u8, leaf_arch_name, "LSTM"))
        .lstm
    else if (std.mem.eql(u8, leaf_arch_name, "ConvNet"))
        .convnet
    else if (std.mem.eql(u8, leaf_arch_name, "Linear"))
        .linear
    else if (std.mem.eql(u8, leaf_arch_name, "SlimmableContainer"))
        // A container nested inside a container does not exist upstream.
        return Error.UnsupportedFeature
    else
        return Error.UnsupportedArchitecture;

    const config: Config = switch (arch) {
        .wavenet => .{ .wavenet = try parseWaveNetConfig(arena, leaf_config) },
        .lstm => .{ .lstm = try parseLstmConfig(leaf_config) },
        .convnet => .{ .convnet = try parseConvNetConfig(arena, leaf_config) },
        .linear => .{ .linear = try parseLinearConfig(leaf_config) },
    };

    // Weights: flat float array, consumed strictly in order; validate the
    // exact total up front (upstream's own end-of-array checks are partly
    // debug-only asserts).
    const weights_value = leaf_root.get("weights") orelse return Error.MissingField;
    const weights_array = switch (weights_value) {
        .array => |a| a,
        else => return Error.InvalidField,
    };
    const weights = try arena.alloc(f32, weights_array.items.len);
    for (weights, weights_array.items) |*dst, item| {
        dst.* = switch (item) {
            .float => |v| @floatCast(v),
            .integer => |v| @floatFromInt(v),
            else => return Error.InvalidField,
        };
    }
    const expected = expectedWeightCount(&config);
    if (weights.len != expected) return Error.WeightCountMismatch;

    // Submodel rates must agree with the container's, -1 tolerated on
    // either side (container.cpp:35-46).
    const leaf_rate = try optionalSampleRate(leaf_root);
    if (submodel_info != null and container_rate > 0 and leaf_rate > 0 and container_rate != leaf_rate) {
        return Error.InvalidField;
    }
    const sample_rate: f64 = if (leaf_rate > 0) leaf_rate else container_rate;

    var metadata = Metadata{};
    if (root.get("metadata")) |meta_value| {
        if (meta_value == .object) {
            const meta = meta_value.object;
            metadata.loudness = optionalNumber(meta.get("loudness"));
            metadata.input_level_dbu = optionalNumber(meta.get("input_level_dbu"));
            metadata.output_level_dbu = optionalNumber(meta.get("output_level_dbu"));
            metadata.gear_type = optionalString(meta.get("gear_type"));
        }
        // null / absent metadata is legal (get_dsp.cpp:152).
    }

    return .{
        .parsed = parsed,
        .raw_bytes = bytes,
        .raw_allocator = allocator,
        .version = version,
        .partial_support = partial_support,
        .architecture = arch,
        .config = config,
        .weights = weights,
        .sample_rate = sample_rate,
        .metadata = metadata,
        .submodel_info = submodel_info,
    };
}

fn optionalSampleRate(root: std.json.ObjectMap) Error!f64 {
    const v = root.get("sample_rate") orelse return -1.0;
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        .null => -1.0,
        else => Error.InvalidField,
    };
}

// ---------------------------------------------------------------------------
// Per-architecture config parsing (incl. the legacy spellings a reader must
// accept; spec §6.1.1/§6.2/§6.3/§6.4 + gotcha 10).
// ---------------------------------------------------------------------------

fn parseWaveNetConfig(arena: std.mem.Allocator, config: std.json.ObjectMap) anyerror!WaveNetConfig {
    const condition_dsp = try parseOptionalConditionDsp(arena, config.get("condition_dsp"));

    const layers_value = config.get("layers") orelse return Error.MissingField;
    const layers_array = switch (layers_value) {
        .array => |a| a,
        else => return Error.InvalidField,
    };
    if (layers_array.items.len == 0) return Error.InvalidField;

    const layers = try arena.alloc(WaveNetLayerArray, layers_array.items.len);
    for (layers, layers_array.items) |*dst, item| {
        const layer_object = switch (item) {
            .object => |o| o,
            else => return Error.InvalidField,
        };
        dst.* = try parseLayerArray(arena, layer_object);
    }

    // Chaining constraints (model.cpp:604-611): the main stream feeds the
    // next array's rechannel (input_size == prev channels) while the head
    // stream requires the next array's residual width to match the previous
    // head output.
    for (layers[1..], layers[0 .. layers.len - 1]) |next, prev| {
        if (next.input_size != prev.channels) return Error.InvalidField;
        if (next.channels != prev.head_out) return Error.InvalidField;
    }

    var post_head: ?WaveNetPostHead = null;
    if (config.get("head")) |head_value| {
        switch (head_value) {
            .null => {},
            .object => |head_object| {
                const kernel_sizes = try parseUsizeArray(arena, head_object.get("kernel_sizes") orelse return Error.MissingField);
                if (kernel_sizes.len == 0) return Error.InvalidField;
                post_head = .{
                    .channels = try requireUsize(head_object, "channels"),
                    .out_channels = try requireUsize(head_object, "out_channels"),
                    .kernel_sizes = kernel_sizes,
                    .activation = try parseActivation(arena, head_object.get("activation") orelse return Error.MissingField),
                };
                // Legacy in_channels, when present, must match the last
                // array's head size (model.cpp:1167-1177).
                if (head_object.get("in_channels")) |v| {
                    if (v != .null and try valueUsize(v) != layers[layers.len - 1].head_out) return Error.InvalidField;
                }
            },
            else => return Error.InvalidField,
        }
    }

    return .{
        .layers = layers,
        .head = post_head,
        .head_scale = @floatCast(try requireNumber(config, "head_scale")),
        .in_channels = try optionalUsize(config, "in_channels", 1),
        .condition_dsp = condition_dsp,
    };
}

fn parseOptionalConditionDsp(arena: std.mem.Allocator, value: ?std.json.Value) anyerror!?*const ConditionDsp {
    const v = value orelse return null;
    return switch (v) {
        .null => null,
        .object => |o| try parseConditionDsp(arena, o),
        else => Error.InvalidField,
    };
}

fn parseConditionDsp(arena: std.mem.Allocator, root: std.json.ObjectMap) anyerror!*const ConditionDsp {
    const version_value = root.get("version") orelse return Error.MissingField;
    const version = try Version.parse(switch (version_value) {
        .string => |s| s,
        else => return Error.InvalidField,
    });
    if (version.lessThan(earliest_version)) return Error.UnsupportedVersion;
    if (version.major > latest_version.major or version.minor > latest_version.minor) return Error.UnsupportedVersion;

    const arch_name = switch (root.get("architecture") orelse return Error.MissingField) {
        .string => |s| s,
        else => return Error.InvalidField,
    };
    const arch: Arch = if (std.mem.eql(u8, arch_name, "WaveNet"))
        .wavenet
    else if (std.mem.eql(u8, arch_name, "LSTM"))
        .lstm
    else if (std.mem.eql(u8, arch_name, "ConvNet"))
        .convnet
    else if (std.mem.eql(u8, arch_name, "Linear"))
        .linear
    else
        return Error.UnsupportedArchitecture;

    const config_object = switch (root.get("config") orelse return Error.MissingField) {
        .object => |o| o,
        else => return Error.InvalidField,
    };
    const config: Config = switch (arch) {
        .wavenet => .{ .wavenet = try parseWaveNetConfig(arena, config_object) },
        .lstm => .{ .lstm = try parseLstmConfig(config_object) },
        .convnet => .{ .convnet = try parseConvNetConfig(arena, config_object) },
        .linear => .{ .linear = try parseLinearConfig(config_object) },
    };

    const weights_value = root.get("weights") orelse return Error.MissingField;
    const weights_array = switch (weights_value) {
        .array => |a| a,
        else => return Error.InvalidField,
    };
    const weights = try arena.alloc(f32, weights_array.items.len);
    for (weights, weights_array.items) |*dst, item| {
        dst.* = switch (item) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => return Error.InvalidField,
        };
    }
    if (weights.len != expectedWeightCount(&config)) return Error.WeightCountMismatch;

    const out = try arena.create(ConditionDsp);
    out.* = .{
        .architecture = arch,
        .config = config,
        .weights = weights,
        .sample_rate = try optionalSampleRate(root),
    };
    return out;
}

fn parseLayerArray(arena: std.mem.Allocator, layer: std.json.ObjectMap) !WaveNetLayerArray {
    // Slimmable arrays reroute to a different runtime upstream — unsupported.
    if (layer.get("slimmable")) |v| {
        if (v != .null) return Error.UnsupportedFeature;
    }
    const groups_input = try optionalUsize(layer, "groups_input", 1);
    const groups_input_mixin = try optionalUsize(layer, "groups_input_mixin", 1);

    const input_size = try requireUsize(layer, "input_size");
    const condition_size = try requireUsize(layer, "condition_size");
    const channels = try requireUsize(layer, "channels");
    const bottleneck = try optionalUsize(layer, "bottleneck", channels);
    if (groups_input == 0 or groups_input_mixin == 0) return Error.InvalidField;
    if (channels % groups_input != 0) return Error.InvalidField;
    if (condition_size % groups_input_mixin != 0) return Error.InvalidField;
    if (bottleneck % groups_input != 0 or bottleneck % groups_input_mixin != 0) return Error.InvalidField;
    const dilations = try parseUsizeArray(arena, layer.get("dilations") orelse return Error.MissingField);
    if (dilations.len == 0) return Error.InvalidField;
    const layer_count = dilations.len;

    // kernel_size (scalar, legacy) XOR kernel_sizes (per layer) — both is an
    // error upstream (model.cpp:919-923).
    const has_scalar_kernel = layer.get("kernel_size") != null;
    const has_kernel_list = layer.get("kernel_sizes") != null;
    if (has_scalar_kernel and has_kernel_list) return Error.InvalidField;
    var kernel_sizes: []usize = undefined;
    if (has_kernel_list) {
        kernel_sizes = try parseUsizeArray(arena, layer.get("kernel_sizes").?);
        if (kernel_sizes.len != layer_count) return Error.InvalidField;
    } else if (has_scalar_kernel) {
        const k = try valueUsize(layer.get("kernel_size").?);
        kernel_sizes = try arena.alloc(usize, layer_count);
        @memset(kernel_sizes, k);
    } else return Error.MissingField;
    // 64 is far above any real NAM config (packed easy mode peaks at 16)
    // and is the streaming engine's tap ceiling.
    for (kernel_sizes) |k| if (k == 0 or k > 64) return Error.InvalidField;

    // activation: scalar or per-layer array.
    const activations = try parseActivationPerLayer(arena, layer.get("activation") orelse return Error.MissingField, layer_count);

    // gating_mode (string or array) / legacy boolean `gated`
    // (model.cpp:991-1095).
    const gating_modes = try arena.alloc(GatingMode, layer_count);
    @memset(gating_modes, .none);
    if (layer.get("gating_mode")) |v| {
        switch (v) {
            .string => |s| @memset(gating_modes, try parseGatingMode(s)),
            .array => |a| {
                if (a.items.len != layer_count) return Error.InvalidField;
                for (gating_modes, a.items) |*dst, item| {
                    const s = switch (item) {
                        .string => |s| s,
                        else => return Error.InvalidField,
                    };
                    dst.* = try parseGatingMode(s);
                }
            },
            else => return Error.InvalidField,
        }
    } else if (layer.get("gated")) |v| {
        const gated = switch (v) {
            .bool => |b| b,
            else => return Error.InvalidField,
        };
        if (gated) @memset(gating_modes, .gated);
    }

    // Secondary activation defaults to Sigmoid for gated layers
    // (model.cpp:1004-1029).
    var secondary: []Activation = undefined;
    if (layer.get("secondary_activation")) |v| {
        secondary = try parseActivationPerLayer(arena, v, layer_count);
    } else {
        secondary = try arena.alloc(Activation, layer_count);
        @memset(secondary, Activation.sigmoid_default);
    }

    // head: new nested form or legacy head_size/head_bias (kernel 1)
    // (model.cpp:883-905).
    var head_out: usize = undefined;
    var head_kernel: usize = undefined;
    var head_bias: bool = undefined;
    if (layer.get("head")) |head_value| {
        const head_object = switch (head_value) {
            .object => |o| o,
            else => return Error.InvalidField,
        };
        head_out = try requireUsize(head_object, "out_channels");
        head_kernel = try requireUsize(head_object, "kernel_size");
        head_bias = try requireBool(head_object, "bias");
    } else if (layer.get("head_size")) |v| {
        head_out = try valueUsize(v);
        head_kernel = 1;
        head_bias = if (layer.get("head_bias")) |b| try requireBoolValue(b) else false;
    } else return Error.MissingField;
    if (head_kernel == 0) return Error.InvalidField;

    // layer1x1 {active, groups}, default {true, 1} (model.cpp:865-873).
    var layer1x1_active = true;
    var layer1x1_groups: usize = 1;
    if (layer.get("layer1x1")) |v| {
        const o = switch (v) {
            .object => |o| o,
            else => return Error.InvalidField,
        };
        layer1x1_active = try requireBool(o, "active");
        layer1x1_groups = try requireUsize(o, "groups");
    }
    if (layer1x1_groups == 0) return Error.InvalidField;
    if (bottleneck % layer1x1_groups != 0 or channels % layer1x1_groups != 0) return Error.InvalidField;
    // layer1x1 inactive requires bottleneck == channels (detail.h:59-65).
    if (!layer1x1_active and bottleneck != channels) return Error.InvalidField;

    // head1x1 {active, out_channels, groups}, default inactive
    // (model.cpp:1104-1114).
    var head1x1_active = false;
    var head1x1_out: usize = channels;
    var head1x1_groups: usize = 1;
    if (layer.get("head1x1")) |v| {
        const o = switch (v) {
            .object => |o| o,
            else => return Error.InvalidField,
        };
        head1x1_active = try requireBool(o, "active");
        head1x1_out = try requireUsize(o, "out_channels");
        head1x1_groups = try requireUsize(o, "groups");
    }
    if (head1x1_groups == 0) return Error.InvalidField;
    if (bottleneck % head1x1_groups != 0 or head1x1_out % head1x1_groups != 0) return Error.InvalidField;

    // The eight FiLM slots (model.cpp:1117-1137): absent | false | object
    // {active?:bool=true, shift?:bool=true, groups?:int=1}.
    const conv_pre_film = try parseFilmParams(layer.get("conv_pre_film"));
    const conv_post_film = try parseFilmParams(layer.get("conv_post_film"));
    const input_mixin_pre_film = try parseFilmParams(layer.get("input_mixin_pre_film"));
    const input_mixin_post_film = try parseFilmParams(layer.get("input_mixin_post_film"));
    const activation_pre_film = try parseFilmParams(layer.get("activation_pre_film"));
    const activation_post_film = try parseFilmParams(layer.get("activation_post_film"));
    const layer1x1_post_film = try parseFilmParams(layer.get("layer1x1_post_film"));
    const head1x1_post_film = try parseFilmParams(layer.get("head1x1_post_film"));
    if (layer1x1_post_film.active and !layer1x1_active) return Error.InvalidField;
    try validateFilmParams(condition_size, channels, conv_pre_film);
    try validateFilmParams(condition_size, condition_size, input_mixin_pre_film);
    try validateFilmParams(condition_size, bottleneck, activation_post_film);
    try validateFilmParams(condition_size, channels, layer1x1_post_film);
    try validateFilmParams(condition_size, head1x1_out, head1x1_post_film);
    for (0..layer_count) |l| {
        const bg = switch (gating_modes[l]) {
            .none => bottleneck,
            .gated, .blended => 2 * bottleneck,
        };
        try validateFilmParams(condition_size, bg, conv_post_film);
        try validateFilmParams(condition_size, bg, input_mixin_post_film);
        try validateFilmParams(condition_size, bg, activation_pre_film);
    }

    return .{
        .input_size = input_size,
        .condition_size = condition_size,
        .channels = channels,
        .bottleneck = bottleneck,
        .head_out = head_out,
        .head_kernel = head_kernel,
        .head_bias = head_bias,
        .dilations = dilations,
        .kernel_sizes = kernel_sizes,
        .activations = activations,
        .gating_modes = gating_modes,
        .secondary_activations = secondary,
        .layer1x1_active = layer1x1_active,
        .layer1x1_groups = layer1x1_groups,
        .head1x1_active = head1x1_active,
        .head1x1_out = head1x1_out,
        .head1x1_groups = head1x1_groups,
        .groups_input = groups_input,
        .groups_input_mixin = groups_input_mixin,
        .conv_pre_film = conv_pre_film,
        .conv_post_film = conv_post_film,
        .input_mixin_pre_film = input_mixin_pre_film,
        .input_mixin_post_film = input_mixin_post_film,
        .activation_pre_film = activation_pre_film,
        .activation_post_film = activation_post_film,
        .layer1x1_post_film = layer1x1_post_film,
        .head1x1_post_film = head1x1_post_film,
    };
}

fn parseFilmParams(value: ?std.json.Value) Error!FiLMParams {
    const v = value orelse return .{};
    return switch (v) {
        .null => .{},
        .bool => |b| if (b) .{ .active = true, .shift = true, .groups = 1 } else .{},
        .object => |o| blk: {
            const active = if (o.get("active")) |a| try requireBoolValue(a) else true;
            const shift = if (o.get("shift")) |s| try requireBoolValue(s) else true;
            const groups = try optionalUsize(o, "groups", 1);
            if (groups == 0) return Error.InvalidField;
            break :blk .{ .active = active, .shift = shift, .groups = groups };
        },
        else => Error.InvalidField,
    };
}

fn validateFilmParams(condition_dim: usize, input_dim: usize, params: FiLMParams) Error!void {
    if (!params.active) return;
    const out_dim = std.math.mul(usize, input_dim, if (params.shift) @as(usize, 2) else 1) catch return Error.InvalidField;
    if (condition_dim % params.groups != 0 or out_dim % params.groups != 0) return Error.InvalidField;
}

fn parseGatingMode(text: []const u8) Error!GatingMode {
    if (std.mem.eql(u8, text, "none")) return .none;
    if (std.mem.eql(u8, text, "gated")) return .gated;
    if (std.mem.eql(u8, text, "blended")) return .blended;
    return Error.InvalidField;
}

fn parseLstmConfig(config: std.json.ObjectMap) !LstmConfig {
    const num_layers = try requireUsize(config, "num_layers");
    const hidden_size = try requireUsize(config, "hidden_size");
    // The engine indexes cells[num_layers-1] and sizes per-layer state, so a
    // zero-layer (or zero-width) LSTM would underflow/OOB at run.
    if (num_layers == 0 or hidden_size == 0) return Error.InvalidField;
    return .{
        .input_size = try requireUsize(config, "input_size"),
        .hidden_size = hidden_size,
        .num_layers = num_layers,
        .in_channels = try optionalUsize(config, "in_channels", 1),
        .out_channels = try optionalUsize(config, "out_channels", 1),
    };
}

fn parseConvNetConfig(arena: std.mem.Allocator, config: std.json.ObjectMap) !ConvNetConfig {
    if (try optionalUsize(config, "groups", 1) != 1) return Error.UnsupportedFeature;
    const dilations = try parseUsizeArray(arena, config.get("dilations") orelse return Error.MissingField);
    if (dilations.len == 0) return Error.InvalidField;
    return .{
        .channels = try requireUsize(config, "channels"),
        .dilations = dilations,
        .batchnorm = try requireBool(config, "batchnorm"),
        .activation = try parseActivation(arena, config.get("activation") orelse return Error.MissingField),
        .in_channels = try optionalUsize(config, "in_channels", 1),
        .out_channels = try optionalUsize(config, "out_channels", 1),
    };
}

fn parseLinearConfig(config: std.json.ObjectMap) !LinearConfig {
    return .{
        .receptive_field = try requireUsize(config, "receptive_field"),
        .bias = try requireBool(config, "bias"),
        .in_channels = try optionalUsize(config, "in_channels", 1),
        .out_channels = try optionalUsize(config, "out_channels", 1),
    };
}

// ---------------------------------------------------------------------------
// Activation config (spec §6.6: string or {type, ...} object; params parsed
// only for PReLU / LeakyReLU / LeakyHardtanh; Hardtanh min/max ignored
// exactly as the C++ parser does).
// ---------------------------------------------------------------------------

fn parseActivation(arena: std.mem.Allocator, value: std.json.Value) !Activation {
    switch (value) {
        .string => |name| return if (name.len == 0) Activation.sigmoid_default else .{ .kind = try activationKind(name) },
        .object => |o| {
            const type_value = o.get("type") orelse return Error.MissingField;
            const name = switch (type_value) {
                .string => |s| s,
                else => return Error.InvalidField,
            };
            var act = Activation{ .kind = try activationKind(name) };
            switch (act.kind) {
                .prelu => {
                    if (o.get("negative_slopes")) |v| {
                        const slopes = switch (v) {
                            .array => |a| a,
                            else => return Error.InvalidField,
                        };
                        const out = try arena.alloc(f32, slopes.items.len);
                        for (out, slopes.items) |*dst, item| dst.* = @floatCast(try numberValue(item));
                        act.negative_slopes = out;
                    } else if (o.get("negative_slope")) |v| {
                        act.negative_slope = @floatCast(try numberValue(v));
                    }
                },
                .leaky_relu => {
                    if (o.get("negative_slope")) |v| act.negative_slope = @floatCast(try numberValue(v));
                },
                .leaky_hardtanh => {
                    if (o.get("min_val")) |v| act.min_val = @floatCast(try numberValue(v));
                    if (o.get("max_val")) |v| act.max_val = @floatCast(try numberValue(v));
                    if (o.get("min_slope")) |v| act.min_slope = @floatCast(try numberValue(v));
                    if (o.get("max_slope")) |v| act.max_slope = @floatCast(try numberValue(v));
                },
                else => {},
            }
            return act;
        },
        else => return Error.InvalidField,
    }
}

fn parseActivationPerLayer(arena: std.mem.Allocator, value: std.json.Value, layer_count: usize) ![]Activation {
    const out = try arena.alloc(Activation, layer_count);
    switch (value) {
        .array => |a| {
            if (a.items.len != layer_count) return Error.InvalidField;
            for (out, a.items) |*dst, item| {
                // The Python exporter writes null secondary_activation
                // entries for ungated layers; the C++ never parses them.
                // Default = Sigmoid (the gated default).
                dst.* = if (item == .null) Activation.sigmoid_default else try parseActivation(arena, item);
            }
        },
        .null => @memset(out, Activation.sigmoid_default),
        else => {
            const act = try parseActivation(arena, value);
            @memset(out, act);
        },
    }
    return out;
}

fn activationKind(name: []const u8) Error!Activation.Kind {
    const map = .{
        .{ "Tanh", Activation.Kind.tanh },
        .{ "Hardtanh", Activation.Kind.hardtanh },
        .{ "Fasttanh", Activation.Kind.fasttanh },
        .{ "ReLU", Activation.Kind.relu },
        .{ "LeakyReLU", Activation.Kind.leaky_relu },
        .{ "PReLU", Activation.Kind.prelu },
        .{ "Sigmoid", Activation.Kind.sigmoid },
        .{ "SiLU", Activation.Kind.silu },
        .{ "Hardswish", Activation.Kind.hardswish },
        .{ "LeakyHardtanh", Activation.Kind.leaky_hardtanh },
        .{ "LeakyHardTanh", Activation.Kind.leaky_hardtanh }, // accepted alias (activations.cpp:72)
        .{ "Softsign", Activation.Kind.softsign },
    };
    inline for (map) |entry| {
        if (std.mem.eql(u8, name, entry[0])) return entry[1];
    }
    return Error.UnsupportedFeature;
}

// ---------------------------------------------------------------------------
// Expected flat-weight counts (spec §6.1.3/§6.2/§6.3/§6.4; validated upstream
// against every example file).
// ---------------------------------------------------------------------------

pub fn expectedWeightCount(config: *const Config) usize {
    return switch (config.*) {
        .wavenet => |*c| waveNetWeightCount(c),
        .lstm => |*c| lstmWeightCount(c),
        .convnet => |*c| convNetWeightCount(c),
        .linear => |*c| saturatingAdd(c.receptive_field, @intFromBool(c.bias)),
    };
}

fn waveNetWeightCount(config: *const WaveNetConfig) usize {
    var total: usize = 0;
    for (config.layers) |*array| {
        const c = array.channels;
        const b = array.bottleneck;
        const dc = array.condition_size;
        total = saturatingAdd(total, saturatingMul(c, array.input_size)); // rechannel, no bias
        for (0..array.layerCount()) |l| {
            const bg = array.gateWidth(l);
            total = saturatingAdd(total, saturatingAdd(saturatingMul(saturatingMul(bg, c / array.groups_input), array.kernel_sizes[l]), bg)); // dilated grouped conv + bias
            total = saturatingAdd(total, saturatingMul(bg, dc / array.groups_input_mixin)); // grouped input mixin, no bias
            if (array.layer1x1_active) total = saturatingAdd(total, saturatingAdd(saturatingMul(c, b / array.layer1x1_groups), c));
            if (array.head1x1_active) total = saturatingAdd(total, saturatingAdd(saturatingMul(array.head1x1_out, b / array.head1x1_groups), array.head1x1_out));
            total = saturatingAdd(total, filmWeightCount(dc, c, array.conv_pre_film));
            total = saturatingAdd(total, filmWeightCount(dc, bg, array.conv_post_film));
            total = saturatingAdd(total, filmWeightCount(dc, dc, array.input_mixin_pre_film));
            total = saturatingAdd(total, filmWeightCount(dc, bg, array.input_mixin_post_film));
            total = saturatingAdd(total, filmWeightCount(dc, bg, array.activation_pre_film));
            total = saturatingAdd(total, filmWeightCount(dc, b, array.activation_post_film));
            total = saturatingAdd(total, filmWeightCount(dc, c, array.layer1x1_post_film));
            total = saturatingAdd(total, filmWeightCount(dc, array.head1x1_out, array.head1x1_post_film));
        }
        const head_in = if (array.head1x1_active) array.head1x1_out else b;
        total = saturatingAdd(total, saturatingMul(saturatingMul(array.head_out, head_in), array.head_kernel));
        if (array.head_bias) total = saturatingAdd(total, array.head_out);
    }
    if (config.head) |*head| {
        var cin = config.layers[config.layers.len - 1].head_out;
        for (head.kernel_sizes, 0..) |k, i| {
            const cout = if (i == head.kernel_sizes.len - 1) head.out_channels else head.channels;
            total = saturatingAdd(total, saturatingAdd(saturatingMul(saturatingMul(cout, cin), k), cout)); // bias always true
            cin = cout;
        }
    }
    return saturatingAdd(total, 1); // head_scale is the final float
}

fn filmWeightCount(condition_dim: usize, input_dim: usize, params: FiLMParams) usize {
    if (!params.active) return 0;
    const out_dim = saturatingMul(input_dim, if (params.shift) @as(usize, 2) else 1);
    return saturatingAdd(saturatingMul(out_dim, condition_dim / params.groups), out_dim);
}

fn lstmWeightCount(config: *const LstmConfig) usize {
    const h = config.hidden_size;
    var total: usize = 0;
    for (0..config.num_layers) |l| {
        const in_l = if (l == 0) config.input_size else h;
        total = saturatingAdd(total, saturatingMul(saturatingMul(4, h), saturatingAdd(in_l, h))); // [W_ih | W_hh] row-major
        total = saturatingAdd(total, saturatingMul(4, h)); // bias (sum of PyTorch's two)
        total = saturatingAdd(total, saturatingMul(2, h)); // initial hidden + cell state
    }
    return saturatingAdd(saturatingAdd(total, saturatingMul(config.out_channels, h)), config.out_channels); // head
}

fn convNetWeightCount(config: *const ConvNetConfig) usize {
    var total: usize = 0;
    var cin = config.in_channels;
    for (config.dilations) |_| {
        const cout = config.channels;
        total = saturatingAdd(total, saturatingMul(saturatingMul(cout, cin), ConvNetConfig.kernel_size));
        if (config.batchnorm) {
            total = saturatingAdd(total, saturatingAdd(saturatingMul(4, cout), 1)); // running_mean, running_var, gamma, beta, eps
        } else {
            total = saturatingAdd(total, cout); // conv bias only without batchnorm
        }
        cin = cout;
    }
    return saturatingAdd(saturatingAdd(total, saturatingMul(config.out_channels, config.channels)), config.out_channels); // head w + b
}

fn saturatingAdd(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}

fn saturatingMul(a: usize, b: usize) usize {
    return std.math.mul(usize, a, b) catch std.math.maxInt(usize);
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

fn numberValue(value: std.json.Value) Error!f64 {
    return switch (value) {
        .float => |v| v,
        .integer => |v| @floatFromInt(v),
        else => Error.InvalidField,
    };
}

fn optionalNumber(value: ?std.json.Value) ?f64 {
    const v = value orelse return null;
    return numberValue(v) catch null;
}

/// A JSON string value, or null for absent / JSON-null / non-string (tolerant,
/// like optionalNumber). Borrows the parse arena.
fn optionalString(value: ?std.json.Value) ?[]const u8 {
    const v = value orelse return null;
    return switch (v) {
        .string => |s| s,
        else => null,
    };
}

fn valueUsize(value: std.json.Value) Error!usize {
    return switch (value) {
        .integer => |v| if (v < 0) Error.InvalidField else @intCast(v),
        .float => |v| {
            // Untrusted .nam input: reject non-finite (inf/nan), negative, and
            // non-integer floats, and anything past i64 range, BEFORE
            // @intFromFloat (which panics when the value is out of range).
            if (!std.math.isFinite(v) or v < 0 or v != @trunc(v)) return Error.InvalidField;
            if (v > @as(f64, @floatFromInt(std.math.maxInt(i64)))) return Error.InvalidField;
            return @intFromFloat(v);
        },
        else => Error.InvalidField,
    };
}

fn requireUsize(object: std.json.ObjectMap, key: []const u8) Error!usize {
    return valueUsize(object.get(key) orelse return Error.MissingField);
}

fn optionalUsize(object: std.json.ObjectMap, key: []const u8, default: usize) Error!usize {
    const v = object.get(key) orelse return default;
    if (v == .null) return default;
    return valueUsize(v);
}

test "valueUsize rejects non-finite, negative, non-integer, and out-of-range floats" {
    try std.testing.expectError(Error.InvalidField, valueUsize(.{ .float = 1e100 }));
    try std.testing.expectError(Error.InvalidField, valueUsize(.{ .float = std.math.inf(f64) }));
    try std.testing.expectError(Error.InvalidField, valueUsize(.{ .float = std.math.nan(f64) }));
    try std.testing.expectError(Error.InvalidField, valueUsize(.{ .float = -1.0 }));
    try std.testing.expectError(Error.InvalidField, valueUsize(.{ .float = 1.5 }));
    try std.testing.expectError(Error.InvalidField, valueUsize(.{ .integer = -1 }));
    try std.testing.expectEqual(@as(usize, 3), try valueUsize(.{ .float = 3.0 }));
    try std.testing.expectEqual(@as(usize, 7), try valueUsize(.{ .integer = 7 }));
}

fn requireNumber(object: std.json.ObjectMap, key: []const u8) Error!f64 {
    return numberValue(object.get(key) orelse return Error.MissingField);
}

fn requireBoolValue(value: std.json.Value) Error!bool {
    return switch (value) {
        .bool => |b| b,
        else => Error.InvalidField,
    };
}

fn requireBool(object: std.json.ObjectMap, key: []const u8) Error!bool {
    return requireBoolValue(object.get(key) orelse return Error.MissingField);
}

fn parseUsizeArray(arena: std.mem.Allocator, value: std.json.Value) ![]usize {
    const array = switch (value) {
        .array => |a| a,
        else => return Error.InvalidField,
    };
    const out = try arena.alloc(usize, array.items.len);
    for (out, array.items) |*dst, item| dst.* = try valueUsize(item);
    return out;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test {
    _ = @import("nam_file_tests.zig");
}
