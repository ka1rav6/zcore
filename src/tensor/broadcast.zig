const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;

const BroadcastError = error{BroadcastNotPossible};

/// Computes the broadcasted shape of two shapes (numpy-style broadcasting).
/// Right-aligns shapes, pads with 1s then takes element-wise max if either element is 1
/// else returns BroadcastError
/// Caller owns the returned slice.
pub fn broadcastShape(allocator: std.mem.Allocator, shapeA: []const usize, shapeB: []const usize) BroadcastError![]usize {
    const ndim = @max(shapeA.len, shapeB.len);
    var result = try allocator.alloc(usize, ndim);
    errdefer allocator.free(result);

    const offA = ndim - shapeA.len;
    const offB = ndim - shapeB.len;

    var i: usize = ndim;
    while (i > 0) {
        i -= 1;
        const dimA = if (i >= offA) shapeA[i - offA] else 1;
        const dimB = if (i >= offB) shapeB[i - offB] else 1;

        if (dimA != dimB and dimA != 1 and dimB != 1)
            return BroadcastError.BroadcastNotPossible;

        result[i] = @max(dimA, dimB);
    }
    return result;
}

/// Broadcast two tensors to their common broadcasted shape.
/// Returns a tuple of broadcasted views (no data copy).
pub fn broadcast(comptime T: type, allocator: std.mem.Allocator, tenA: *const Tensor(T), tenB: *const Tensor(T)) !struct { Tensor(T), Tensor(T) } {
    const shape = try broadcastShape(allocator, tenA.get_shape(), tenB.get_shape());
    defer allocator.free(shape);

    const a = try tenA.*.broadcast_to(shape);
    const b = try tenB.*.broadcast_to(shape);
    return .{ a, b };
}
