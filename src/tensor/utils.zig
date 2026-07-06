const std = @import("std");

// NOTE: planning on storing data in tensors in flattened row-major format

/// Computes the strides, based on the shape, required to move around the tensor
pub fn computeStrides(shape: []const usize, strides: []usize) void {
    // to see how strides actually work, check the "computeStrides" test
    if (shape.len == 0) return;
    var i = shape.len - 1;
    strides[i] = 1;
    while (i > 0) {
        i -= 1;
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

/// returns the total number of elements inside the tensor
/// basically _data.size();
pub fn numElements(shape: []const usize) usize {
    var total: usize = 1;
    for (shape) |d| { // d = dimension
        total *= d;
    }
    return total;
}
