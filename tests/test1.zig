const Tensor = @import("zcore").Tensor;
const std    = @import("std");

test "completeTest1" {
    var myTensor = try Tensor(i8).init(std.heap.page_allocator, &[_]usize{ 3, 3 });
    defer myTensor.destroy(); // destroys it at the end of the test
    myTensor.fill(10);
    try std.testing.expectEqual(@as(i8, 10), myTensor.get(&[_]usize{ 2, 2 }).*);
}
