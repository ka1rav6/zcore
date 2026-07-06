const std = @import("std");
const Tensor = @import("core.zig").Tensor;
const utils = @import("utils.zig");
const computeStrides = utils.computeStrides;
const numElements = utils.numElements;

test computeStrides {
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

test numElements {
    // checking if numElements are correct
    std.testing.expectEqual(numElements([_]usize{3, 3}), 9);
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
    t.set([_]usize{2, 2}, 10);
    std.testing.expectEqual(t.get([_]usize{2, 2}), 10);

}

// -------------------------------- Trying different debug prints ------------------------------------
test "Tensor.debugPrint.0d" {
    // debugPrint should not crash for a 0-dimensional tensor (scalar)
    const allocator = std.testing.allocator;
    const shape = [_]usize{};
    var t = try Tensor(u32).init(allocator, shape[0..]);
    defer t.destroy();
    t.debugPrint();
}

test "Tensor.debugPrint.1d" {
    // debugPrint should not crash for a 1-dimensional tensor
    const allocator = std.testing.allocator;
    const shape = [_]usize{ 4 };
    var t = try Tensor(u32).zeroes(allocator, shape[0..]);
    defer t.destroy();
    t.set([_]usize{0}, 10);
    t.set([_]usize{1}, 20);
    t.set([_]usize{2}, 30);
    t.set([_]usize{3}, 40);
    t.debugPrint();
}

test "Tensor.debugPrint.2d" {
    // debugPrint should not crash for a 2-dimensional tensor
    const allocator = std.testing.allocator;
    const shape = [_]usize{ 2, 3 };
    var t = try Tensor(u32).zeroes(allocator, shape[0..]);
    defer t.destroy();
    t.set([_]usize{0, 0}, 1);
    t.set([_]usize{0, 1}, 2);
    t.set([_]usize{0, 2}, 3);
    t.set([_]usize{1, 0}, 4);
    t.set([_]usize{1, 1}, 5);
    t.set([_]usize{1, 2}, 6);
    t.debugPrint();
}

test "Tensor.debugPrint.3d" {
    // debugPrint should not crash for a 3-dimensional tensor
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
    t.debugPrint();
}

// -------------------------- Setting rows/cols/whole tests --------------------------------

// TODO:
test "Tensor.setRow" {


}

test "Tensor.setCol" {

}

test "Tensor.setWhole"{


}
