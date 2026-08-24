//! Tests for the live-loop terminal UI (`ui.zig`): frame composition (level
//! meters, flags, keybindings), UTF-8-safe profile-name truncation, and the
//! purity/diff contract. The frame-composition tests reference non-`pub`
//! internals (`composeFrame`, `default_height`, `clip_on`) and therefore stay
//! inline in `ui.zig`; this file is the discovery sibling for the convention.
const std = @import("std");
const ui = @import("ui.zig");
