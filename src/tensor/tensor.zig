const std = @import("std");

pub const Tensor = @import("core.zig").Tensor;

comptime {
    if (@import("builtin").is_test) {
        _ = @import("tests.zig");
    }
}
