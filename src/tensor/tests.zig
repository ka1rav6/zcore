const std = @import("std");
const Tensor = @import("core.zig").Tensor;
const utils = @import("utils.zig");

test "computeStrides" {
    // example :
    // [[1, 2, 3]
    //  [4, 5, 6]]
    // => in row major == [1, 2, 3, 4, 5, 6]
    // therefore stride = [3, 1] (jumping a row == move front/back thrice and jumping a col => moving once)
    const shape = [_]usize{ 2, 3 };
    var strides = [_]usize{ 0, 0 };

    utils.computeStrides(shape[0..], strides[0..]);
    try std.testing.expectEqual(@as(usize, 3), strides[0]);
    try std.testing.expectEqual(@as(usize, 1), strides[1]);
}

test "zeroes" {
    // the tensor should be filled with zeroes
    const allocator = std.testing.allocator;
    const shape = [_]usize{ 3, 3 };
    var t = try Tensor(u32).zeroes(allocator, shape[0..]);
    defer t.destroy();
    for (t._data) |elem| {
        try std.testing.expectEqual(@as(u32, 0), elem);
    }
}
