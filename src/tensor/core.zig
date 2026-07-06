const std = @import("std");
const utils = @import("utils.zig");

/// A basic creator for the Tensor struct
/// Each Tensor is also given its own allocator
pub fn Tensor(comptime T: type) type {
    return struct {
        // -------------- members ---------------
        _allocator: std.mem.Allocator,
        _data: []T,
        _shape: []usize,
        _strides: []usize,
        _owns_memory: bool = true,

        const Self = @This(); // so methods can directly use 'Self' instead of @This() everywhere

        // --------------------- constructors and destructors --------------------------

        /// The constructor for the tensor struct
        /// requires allocator and shape as input
        pub fn init(allocator: std.mem.Allocator, shape: []const usize) !Self {
            // because we dont own the memory the original shape points to:
            const shape_copy = try allocator.dupe(usize, shape); // duplicates
            // allocating the strides on the heap too
            const strides = try allocator.alloc(usize, shape.len);
            utils.computeStrides(shape_copy, strides);
            const data = try allocator.alloc(T, utils.numElements(shape));

            return Self{
                ._data = data,
                ._allocator = allocator,
                ._shape = shape_copy,
                ._strides = strides,
            };
        }

        /// Destructor for the tensor struct.
        /// Frees data (if owned by the Tensor)
        /// and the shape and strides array as well
        pub fn destroy(self: *Self) void {
            if (self._owns_memory)
                self._allocator.free(self._data);
            self._allocator.free(self._shape);
            self._allocator.free(self._strides);
        }

        // ---------------------------- other methods -------------------------------------

        /// converts multi-dimensional indices into a single index for the flattened _data array
        fn offset(self: Self, indices: []const usize) usize {
            // EXPLAINATION :
            // user wants : [1, 2]
            // but we have flattened array stored
            // so it converts it into the index we want

            std.debug.assert(indices.len == self._shape.len);
            var off: usize = 0;
            for (indices, 0..) |idx, i| {
                off += idx * self._strides[i]; // standard row major calculation
            }
            return off;
        }
        /// returns the value of the index passed by the user
        /// internally calls the offset function to get the index in row-major format
        pub fn get(self: Self, indices: []const usize) *T {
            return &self._data[self.offset(indices)];
        }

        /// fills the whole tensor with the value taken in input
        /// takes O(n) time where n is the size of the data
        pub fn fill(self: *Self, value: T) void {
            for (self._data) |*x| {
                x.* = value;
            }
        }

        /// Initializes the tensor and returns the instance of the tensor
        /// with zeroes filled inside the data by default
        pub fn zeroes(allocator: std.mem.Allocator, shape: []const usize) !Self {
            var temp_t: Self = undefined;
            temp_t = try Self.init(allocator, shape); // calling the constructor
            @memset(temp_t._data, 0);
            return temp_t;
        }

        /// Sets the value of the particular index to the val
        pub fn set(self: *Self, indices: []const usize, val: T) void {
            const idx = self.offset(indices);
            self._data[idx] = val;
        }

        /// Prints the tensor's metadata and data to stderr in a human-readable format.
        ///
        /// Output includes the element type, shape, strides, and the tensor contents
        /// formatted with nested brackets that reflect the tensor's dimensionality.
        /// Example output for a 2D tensor of shape [2, 3]:
        /// Tensor(u32): shape={ 2, 3 }, strides={ 3, 1 }
        /// [[1, 2, 3],
        ///  [4, 5, 6]]
        pub fn debugPrint(self: Self) void {
            const T_name = @typeName(T);
            std.debug.print("Tensor({s}): shape={any}, strides={any}\n", .{ T_name, self._shape, self._strides });

            if (self._shape.len == 0) {
                // 0-dimensional tensor: print the single scalar value
                std.debug.print("{}\n", .{self._data[0]});
                return;
            }

            // Recursive helper to print one dimension of the tensor.
            // We use a static function inside a nested struct so it can call itself
            // while still having access to the comptime type `T` from the outer scope.
            const print_impl = struct {
                fn printDim(dim: usize, base: usize, data: []const T, shape: []const usize, strides: []const usize) void {
                    if (dim == shape.len - 1) {
                        // Innermost dimension: print the actual data values
                        std.debug.print("[", .{});
                        for (0..shape[dim]) |i| {
                            if (i > 0) std.debug.print(", ", .{});
                            std.debug.print("{}", .{data[base + i]});
                        }
                        std.debug.print("]", .{});
                    } else {
                        std.debug.print("[", .{});
                        for (0..shape[dim]) |i| {
                            if (i > 0) {
                                // Start a new row and indent to align with the inner content
                                std.debug.print(",\n", .{});
                                for (0..dim + 1) |_| std.debug.print(" ", .{});
                            }
                            printDim(dim + 1, base + i * strides[dim], data, shape, strides);
                        }
                        std.debug.print("]", .{});
                    }
                }
            }.printDim;

            print_impl(0, 0, self._data, self._shape, self._strides);
            std.debug.print("\n", .{});
        }

        /// Set the particular row of the tensor to the new one
        pub fn setRow(self: *Self, row_num: usize, new_row: []const T) void {
            const n = self._shape.len;
            std.debug.assert(n >= 1);
            std.debug.assert(row_num < self._shape[0]);
            const row_size = self._strides[0];
            std.debug.assert(new_row.len == row_size);
            const offset_start = row_num * row_size;
            for (new_row, 0..) |val, i| {
                self._data[offset_start + i] = val;
            }
        }

        /// Set the particular column of the tensor to the new one
        pub fn setCol(self: *Self, col_num: usize, new_col: []const T) void {
            const n = self._shape.len;
            if (n == 1) { // just replace the column if there is just one column
                std.debug.assert(col_num < self._shape[0]);
                std.debug.assert(new_col.len == 1);
                self._data[col_num] = new_col[0];
                return;
            }
            const outer_dims = self._shape[0 .. n - 1];
            const outer_count = utils.numElements(outer_dims);
            std.debug.assert(col_num < self._shape[n - 1]);
            std.debug.assert(new_col.len == outer_count);
            const col_stride = self._strides[n - 2];
            var off = col_num;
            for (0..outer_count) |i| {
                self._data[off] = new_col[i];
                off += col_stride;
            }
        }

        /// set the whole tensor to the one passed as an array
        /// the array can either be flattened or not
        pub fn setWhole(self: *Self, array: []const T) void {
            std.debug.assert(array.len == self._data.len);
            for (array, 0..) |val, i| {
                self._data[i] = val;
            }
        }
    };
}
