const std = @import("std");

/// A basic creator for the Tensor struct
/// Each Tensor is also given its own allocator
pub fn Tensor(comptime T: type) type {
    return struct {
        // properties:
        _allocator   : std.mem.Allocator,
        _data        : []T              ,
        _shape       : []usize          ,
        _strides     : [] usize         ,
        _owns_memory : bool = true      ,
        
        const Self = @This(); // so methods can directly use 'Self' instead of @This() everywhere

        // NOTE: planning on storing data in tensors in flattened row-major format
        // Hence, strides will be calculated according to shape

        /// Computes the strides required to move around the tensor
        fn computeStrides(shape: []const usize, strides: []usize) void{
            // to see how strides actually work, check the "computeStrides" test
            if (shape.len == 0) return;
            var i = shape.len - 1;
            strides[i] = 1;
            while (i > 0){
                i -= 1;
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
    };
}

test "computeStrides" {
    // example :
    // [[1, 2, 3]
    //  [4, 5, 6]]
    // => in row major == [1, 2, 3, 4, 5, 6]
    // therefore stride = [3, 1] (jumping a row == move front/back thrice and jumping a col => moving once)
    const shape = [_]usize{ 2, 3 };
    var strides = [_]usize{ 0, 0 };

    const TensorU32 = Tensor(u32);
    TensorU32.computeStrides(shape[0..], strides[0..]);
    try std.testing.expectEqual(@as(usize, 3), strides[0]);
    try std.testing.expectEqual(@as(usize, 1), strides[1]);
}
