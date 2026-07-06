const std = @import("std");
const Tensor = @import("core.zig").Tensor;
const utils = @import("utils.zig");
const compute_strides = utils.compute_strides;
const num_elements = utils.num_elements;

test compute_strides {
    // example :
    // [[1, 2, 3]
    //  [4, 5, 6]]
    // => in row major == [1, 2, 3, 4, 5, 6]
    // therefore stride = [3, 1] (jumping a row == move front/back thrice and jumping a col => moving once)
    const shape = [_]usize{ 2, 3 };
    var strides = [_]usize{ 0, 0 };

    utils.compute_strides(shape[0..], strides[0..]);
    try std.testing.expectEqual(@as(usize, 3), strides[0]);
    try std.testing.expectEqual(@as(usize, 1), strides[1]);
}

test num_elements {
    // checking if num_elements are correct
    try std.testing.expectEqual(num_elements(&[_]usize{ 3, 3 }), 9);
}

test "Tensor.zeroes" {
    // the tensor should be filled with zeroes
    const allocator = std.testing.allocator;
    const shape = [_]usize{ 3, 3 };
    var t = try Tensor(u32).zeroes(allocator, shape[0..]);
    defer t.destroy();
    for (t._data) |elem| {
        try std.testing.expectEqual(@as(u32, 0), elem);
    }
}

test "Tensor.set" {
    // to check if the set method of tensors sets a particular element correctly
    const allocator = std.testing.allocator;
    const shape = [_]usize{ 3, 3 };
    var t = try Tensor(f32).zeroes(allocator, shape[0..]);
    defer t.destroy();
    t.set([_]usize{ 2, 2 }, 10);
    try std.testing.expectEqual(@as(f32, 10), t.get(&[_]usize{ 2, 2 }).*);
}

// -------------------------------- Trying different debug prints ------------------------------------
test "Tensor.debug_print.0d" {
    // debug_print should not crash for a 0-dimensional tensor (scalar)
    const allocator = std.testing.allocator;
    const shape = [_]usize{};
    var t = try Tensor(u32).init(allocator, shape[0..]);
    defer t.destroy();
    t.debug_print();
}

test "Tensor.debug_print.1d" {
    // debug_print should not crash for a 1-dimensional tensor
    const allocator = std.testing.allocator;
    const shape = [_]usize{4};
    var t = try Tensor(u32).zeroes(allocator, shape[0..]);
    defer t.destroy();
    t.set([_]usize{0}, 10);
    t.set([_]usize{1}, 20);
    t.set([_]usize{2}, 30);
    t.set([_]usize{3}, 40);
    t.debug_print();
}

test "Tensor.debug_print.2d" {
    // debug_print should not crash for a 2-dimensional tensor
    const allocator = std.testing.allocator;
    const shape = [_]usize{ 2, 3 };
    var t = try Tensor(u32).zeroes(allocator, shape[0..]);
    defer t.destroy();
    t.set([_]usize{ 0, 0 }, 1);
    t.set([_]usize{ 0, 1 }, 2);
    t.set([_]usize{ 0, 2 }, 3);
    t.set([_]usize{ 1, 0 }, 4);
    t.set([_]usize{ 1, 1 }, 5);
    t.set([_]usize{ 1, 2 }, 6);
    t.debug_print();
}

test "Tensor.debug_print.3d" {
    // debug_print should not crash for a 3-dimensional tensor
    const allocator = std.testing.allocator;
    const shape = [_]usize{ 2, 2, 2 };
    var t = try Tensor(u32).zeroes(allocator, shape[0..]);
    defer t.destroy();
    var val: u32 = 1;
    for (0..2) |i| {
        for (0..2) |j| {
            for (0..2) |k| {
                t.set([_]usize{ i, j, k }, val);
                val += 1;
            }
        }
    }
    t.debug_print();
}

// -------------------------- Setting rows/cols/whole tests --------------------------------

// TODO:
test "Tensor.setRow" {}

test "Tensor.setCol" {}

test "Tensor.setWhole" {}
