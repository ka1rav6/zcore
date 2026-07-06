const std    = @import("std");
const Tensor = @import("zcore").Tensor;


pub fn main() !void {
    
    const alloc = std.heap.page_allocator;
    const arr = [_]f16{12.0, 10.0, 5.0,
                        8.0 , 11.0, 0.0,
                        7.0 , -1.0,-10.0};
    var mat1 = try Tensor(f16)
                .init(alloc, &[_]usize{3, 3});
    mat1.setWhole(&arr);

    mat1.debug_print();
}
