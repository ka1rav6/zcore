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

        pub const Error = error{IndexOutOfBounds};

        // --------------------- constructors and destructors --------------------------

        /// The constructor for the tensor struct
        /// requires allocator and shape as input
        pub fn init(allocator: std.mem.Allocator, shape: []const usize) !Self {
            comptime {
                if (@typeInfo(T) != .int and @typeInfo(T) != .float)
                    @compileError("Tensor can only contain numerical types\n");
            }
            // because we dont own the memory the original shape points to:
            const shape_copy = try allocator.dupe(usize, shape); // duplicates
            errdefer allocator.free(shape_copy);
            // allocating the strides on the heap too
            const strides = try allocator.alloc(usize, shape.len);
            errdefer allocator.free(strides);
            utils.compute_strides(shape_copy, strides);
            const data = try allocator.alloc(T, utils.num_elements(shape));

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
        fn offset(self: Self, indices: []const usize) Error!usize {
            if (indices.len != self._shape.len) return error.IndexOutOfBounds;
            var off: usize = 0;
            for (indices, 0..) |idx, i| {
                if (idx >= self._shape[i]) return error.IndexOutOfBounds;
                off += idx * self._strides[i];
            }
            return off;
        }
        /// returns the value at the given indices (bounds-checked)
        pub fn get(self: Self, indices: []const usize) Error!*const T {
            return &self._data[try self.offset(indices)];
        }

        /// returns the value at the given indices (no bounds checking)
        pub fn getUnchecked(self: Self, indices: []const usize) *const T {
            var off: usize = 0;
            for (indices, 0..) |idx, i| {
                off += idx * self._strides[i];
            }
            return &self._data[off];
        }

        /// getter for shape member of tensor struct
        pub fn getShape(self: *Self) []const usize {
            return self._shape;
        }

        /// getter for stride member of tensor struct
        pub fn getStride(self: *Self) []const usize {
            return self._strides;
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

        /// Sets the value at the given indices (bounds-checked)
        pub fn set(self: *Self, indices: []const usize, val: T) Error!void {
            const idx = try self.offset(indices);
            self._data[idx] = val;
        }

        /// Set the particular row of the tensor to the new one
        /// Uses stride-aware iteration so it works correctly with transposed tensors
        pub fn setRow(self: *Self, row_num: usize, new_row: []const T) void {
            const n = self._shape.len;
            // if number of rows is less than 1
            // then you cannot set the row... hence:
            std.debug.assert(n >= 1);
            // checking bounds

            std.debug.assert(row_num < self._shape[0]);
            const row_size = utils.num_elements(self._shape[1..]);
            std.debug.assert(new_row.len == row_size);

            const base = row_num * self._strides[0];
            for (0..row_size) |flat_i| {
                var off = base;
                var remaining = flat_i;
                var d: usize = n;

                while (d > 1) {
                    d -= 1;
                    off += (remaining % self._shape[d]) * self._strides[d];
                    remaining /= self._shape[d];
                }
                self._data[off] = new_row[flat_i];
            }
        }

        /// Set the particular column of the tensor to the new one
        /// Only works for 2D tensors
        pub fn setCol(self: *Self, col_num: usize, new_col: []const T) void {
            std.debug.assert(self._shape.len == 2);
            const rows = self._shape[0];
            const cols = self._shape[1];

            // bound checking
            std.debug.assert(col_num < cols);
            std.debug.assert(new_col.len == rows);
            for (0..rows) |row| {
                const idx = row * self._strides[0] + col_num * self._strides[1];
                self._data[idx] = new_col[row];
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

        /// Resizes the current tensor
        /// New memory is allocated and previous memory is freed
        /// Then the current tensor is mem copied to the new memory as well
        pub fn resize(self: *Self, new_shape: []const usize) !void {
            const new_num = utils.num_elements(new_shape);
            const old_num = self._data.len;
            const copy_len = @min(new_num, old_num);

            const new_data = try self._allocator.alloc(T, new_num);
            errdefer self._allocator.free(new_data);
            @memcpy(new_data[0..copy_len], self._data[0..copy_len]);
            if (new_num > old_num)
                @memset(new_data[copy_len..], 0);

            const shape_copy = try self._allocator.dupe(usize, new_shape);
            errdefer self._allocator.free(shape_copy);
            const new_strides = try self._allocator.alloc(usize, new_shape.len);
            utils.compute_strides(shape_copy, new_strides);
            if (self._owns_memory)
                self._allocator.free(self._data);
            self._allocator.free(self._shape);
            self._allocator.free(self._strides);
            self._data = new_data;
            self._shape = shape_copy;
            self._strides = new_strides;
            self._owns_memory = true;
        }
        /// Transposes the tensor
        /// Doesn't actually copy/ modify the data
        /// We just play with the strides/shapes so now all the functions
        /// effectly help access the elements of the transpose instead
        /// O(1)
        pub fn transpose(self: *Self) void {
            std.debug.assert(self._shape.len == 2);
            const tmp = self._shape[0];
            self._shape[0] = self._shape[1];
            self._shape[1] = tmp;
            const tmp_s = self._strides[0];
            self._strides[0] = self._strides[1];
            self._strides[1] = tmp_s;
        }

        //------------------------------ DEBUG FUNCTIONS -------------------------------

        /// Prints the tensor's metadata and data to stderr in a human-readable format.
        ///
        /// Output includes the element type, shape, strides, and the tensor contents
        /// formatted with nested brackets that reflect the tensor's dimensionality.
        /// Example output for a 2D tensor of shape [2, 3]:
        /// Tensor(u32): shape={ 2, 3 }, strides={ 3, 1 }
        /// [[1, 2, 3],
        ///  [4, 5, 6]]
        pub fn debug_print(self: Self) void {
            const T_name = @typeName(T);
            std.debug.print("Tensor({s}): shape={any}, strides={any}\n", .{ T_name, self._shape, self._strides });
            if (self._data.len == 0) {
                std.debug.print("0 size tensor\n", .{});
                return;
            }
            if (self._shape.len == 0) {
                // 0-dimensional tensor: print the single scalar value
                std.debug.print("{}\n", .{self._data[0]});
                return;
            }

            // Recursive helper to print one dimension of the tensor.
            // We use a static function inside a nested struct so it can call itself
            // while still having access to the comptime type `T` from the outer scope.
            const print_impl = struct {
                fn print_dim(dim: usize, base: usize, data: []const T, shape: []const usize, strides: []const usize) void {
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
                            print_dim(dim + 1, base + i * strides[dim], data, shape, strides);
                        }
                        std.debug.print("]", .{});
                    }
                }
            }.print_dim;
            print_impl(0, 0, self._data, self._shape, self._strides);
            std.debug.print("\n", .{});
        }
    };
}
