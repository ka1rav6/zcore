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
    try t.set([_]usize{ 2, 2 }, 10);
    try std.testing.expectEqual(@as(f32, 10), (try t.get(&[_]usize{ 2, 2 })).*);
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
    try t.set([_]usize{0}, 10);
    try t.set([_]usize{1}, 20);
    try t.set([_]usize{2}, 30);
    try t.set([_]usize{3}, 40);
    t.debug_print();
}

test "Tensor.debug_print.2d" {
    // debug_print should not crash for a 2-dimensional tensor
    const allocator = std.testing.allocator;
    const shape = [_]usize{ 2, 3 };
    var t = try Tensor(u32).zeroes(allocator, shape[0..]);
    defer t.destroy();
    try t.set([_]usize{ 0, 0 }, 1);
    try t.set([_]usize{ 0, 1 }, 2);
    try t.set([_]usize{ 0, 2 }, 3);
    try t.set([_]usize{ 1, 0 }, 4);
    try t.set([_]usize{ 1, 1 }, 5);
    try t.set([_]usize{ 1, 2 }, 6);
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
                try t.set([_]usize{ i, j, k }, val);
                val += 1;
            }
        }
    }
    t.debug_print();
}

// -------------------------- Setting rows/cols/whole tests --------------------------------

test "Tensor.setRow" {
    const allocator = std.testing.allocator;
    const shape = [_]usize{ 3, 3 };
    var t = try Tensor(u32).zeroes(allocator, shape[0..]);
    defer t.destroy();

    const new_row = [_]u32{ 10, 20, 30 };
    t.setRow(1, &new_row); // set row one as the new row

    // row 1 should be updated
    try std.testing.expectEqual(@as(u32, 10), (try t.get(&[_]usize{ 1, 0 })).*);
    try std.testing.expectEqual(@as(u32, 20), (try t.get(&[_]usize{ 1, 1 })).*);
    try std.testing.expectEqual(@as(u32, 30), (try t.get(&[_]usize{ 1, 2 })).*);
    // row 0 should still be zero
    try std.testing.expectEqual(@as(u32, 0), (try t.get(&[_]usize{ 0, 0 })).*);
    // row 2 should still be zero
    try std.testing.expectEqual(@as(u32, 0), (try t.get(&[_]usize{ 2, 0 })).*);
}

test "Tensor.setCol" { // only works for 2d tensors of course
    const allocator = std.testing.allocator;
    const shape = [_]usize{ 3, 3 };
    var t = try Tensor(u32).zeroes(allocator, shape[0..]);
    defer t.destroy();

    const new_col = [_]u32{ 10, 20, 30 };
    t.setCol(1, &new_col);

    // col 1 should be updated
    try std.testing.expectEqual(@as(u32, 10), (try t.get(&[_]usize{ 0, 1 })).*);
    try std.testing.expectEqual(@as(u32, 20), (try t.get(&[_]usize{ 1, 1 })).*);
    try std.testing.expectEqual(@as(u32, 30), (try t.get(&[_]usize{ 2, 1 })).*);
    // col 0 should still be zero
    try std.testing.expectEqual(@as(u32, 0), (try t.get(&[_]usize{ 0, 0 })).*);
    // col 2 should still be zero
    try std.testing.expectEqual(@as(u32, 0), (try t.get(&[_]usize{ 0, 2 })).*);
}

test "Tensor.setWhole" {
    const allocator = std.testing.allocator;
    const shape = [_]usize{ 2, 3 };
    var t = try Tensor(u32).zeroes(allocator, shape[0..]);
    defer t.destroy();

    const values = [_]u32{ 1, 2, 3, 4, 5, 6 };
    t.setWhole(&values);

    // checking each value separately
    try std.testing.expectEqual(@as(u32, 1), (try t.get(&[_]usize{ 0, 0 })).*);
    try std.testing.expectEqual(@as(u32, 2), (try t.get(&[_]usize{ 0, 1 })).*);
    try std.testing.expectEqual(@as(u32, 3), (try t.get(&[_]usize{ 0, 2 })).*);
    try std.testing.expectEqual(@as(u32, 4), (try t.get(&[_]usize{ 1, 0 })).*);
    try std.testing.expectEqual(@as(u32, 5), (try t.get(&[_]usize{ 1, 1 })).*);
    try std.testing.expectEqual(@as(u32, 6), (try t.get(&[_]usize{ 1, 2 })).*);
}

test "Tensor.slice" {
    const allocator = std.testing.allocator;
    const shape = [_]usize{ 3, 4 };
    var t = try Tensor(u32).zeroes(allocator, shape[0..]);
    defer t.destroy();

    // fill with sequential values
    var val: u32 = 0;
    for (0..3) |i| {
        for (0..4) |j| {
            try t.set([_]usize{ i, j }, val);
            val += 1;
        }
    }

    var slice = try t.slice(1, 3);
    defer slice.destroy();

    // shape should be (2, 4)
    try std.testing.expectEqual(@as(usize, 2), slice._shape[0]);
    try std.testing.expectEqual(@as(usize, 4), slice._shape[1]);
    // slice should not own memory
    try std.testing.expectEqual(false, slice._owns_memory);
    // check values in slice
    try std.testing.expectEqual(@as(u32, 4), (try slice.get(&[_]usize{ 0, 0 })).*);
    try std.testing.expectEqual(@as(u32, 5), (try slice.get(&[_]usize{ 0, 1 })).*);
    try std.testing.expectEqual(@as(u32, 8), (try slice.get(&[_]usize{ 1, 0 })).*);
    try std.testing.expectEqual(@as(u32, 11), (try slice.get(&[_]usize{ 1, 3 })).*);
}

test "Tensor.transpose" {
    // Meaning of transpose:
    // if A is a tensor, for example :
    // A = [[1,2,3]
    //      [4,5,6]
    //      [7,8,9]]
    // Then, the transpose of A is:
    //  [[1,4,7]
    //   [2,5,8]
    //   [3,6,9]]

    const allocator = std.testing.allocator;
    const shape = [_]usize{ 2, 3 };
    var t = try Tensor(u32).zeroes(allocator, shape[0..]);
    defer t.destroy();

    // fill with sequential values
    var val: u32 = 1;
    for (0..2) |i| {
        for (0..3) |j| {
            try t.set([_]usize{ i, j }, val);
            val += 1;
        }
    }

    t.transpose();
    // shape should be (3, 2)
    try std.testing.expectEqual(@as(usize, 3), t._shape[0]);
    try std.testing.expectEqual(@as(usize, 2), t._shape[1]);
    // check transposed values
    try std.testing.expectEqual(@as(u32, 1), (try t.get(&[_]usize{ 0, 0 })).*);
    try std.testing.expectEqual(@as(u32, 4), (try t.get(&[_]usize{ 0, 1 })).*);
    try std.testing.expectEqual(@as(u32, 2), (try t.get(&[_]usize{ 1, 0 })).*);
    try std.testing.expectEqual(@as(u32, 5), (try t.get(&[_]usize{ 1, 1 })).*);
    try std.testing.expectEqual(@as(u32, 3), (try t.get(&[_]usize{ 2, 0 })).*);
    try std.testing.expectEqual(@as(u32, 6), (try t.get(&[_]usize{ 2, 1 })).*);
}

test "Tensor.resize" {
    // Resizing includes preserving the data in the old places
    // and making the data of the new places (if any) be 0
    // Example: 2 * 3 -> 3 * 3
    //
    // [[1,2,3]      ->   [[1,2,3]
    //  [4,5,6]]     ->    [4,5,6]
    //                     [0,0,0]]

    const allocator = std.testing.allocator;
    const shape = [_]usize{ 2, 3 };
    var t = try Tensor(u32).zeroes(allocator, shape[0..]);
    defer t.destroy();
    // fill with sequential values
    var val: u32 = 1;
    for (0..2) |i| {
        for (0..3) |j| {
            try t.set([_]usize{ i, j }, val);
            val += 1;
        }
    }
    // resize to larger (2x3 -> 3x3)
    const larger = [_]usize{ 3, 3 };
    try t.resize(&larger);
    try std.testing.expectEqual(@as(usize, 3), t._shape[0]);
    try std.testing.expectEqual(@as(usize, 3), t._shape[1]);

    // old values preserved
    try std.testing.expectEqual(@as(u32, 1), (try t.get(&[_]usize{ 0, 0 })).*);
    try std.testing.expectEqual(@as(u32, 6), (try t.get(&[_]usize{ 1, 2 })).*);
    // new elements zeroed
    try std.testing.expectEqual(@as(u32, 0), (try t.get(&[_]usize{ 2, 0 })).*);
    try std.testing.expectEqual(@as(u32, 0), (try t.get(&[_]usize{ 2, 2 })).*);

    // resize to smaller (3x3 -> 1x2)
    const smaller = [_]usize{ 1, 2 };
    try t.resize(&smaller);
    try std.testing.expectEqual(@as(usize, 1), t._shape[0]);
    try std.testing.expectEqual(@as(usize, 2), t._shape[1]);
    // values preserved
    try std.testing.expectEqual(@as(u32, 1), (try t.get(&[_]usize{ 0, 0 })).*);
    try std.testing.expectEqual(@as(u32, 2), (try t.get(&[_]usize{ 0, 1 })).*);
}

test "Type rejection at comptime for non-numeric types" {
    // All supported numerical types must compile without error
    _ = Tensor(u8);
    _ = Tensor(u32);
    _ = Tensor(u64);
    _ = Tensor(i32);
    _ = Tensor(i64);
    _ = Tensor(f32);
    _ = Tensor(f64);
    _ = Tensor(bool);
}
