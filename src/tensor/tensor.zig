const std = @import("std");


// ------------------------------------ SUPPORTING FUNCTIONS --------------------------------------

// NOTE: planning on storing data in tensors in flattened row-major format

/// Computes the strides, based on the shape, required to move around the tensor
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

/// returns the total number of elements inside the tensor
/// basically _data.size();
fn numElements(shape: []const usize) usize{
    var total: usize = 1;
    for (shape) |d|{ // d = dimension
        total *= d;
    }
    return total; 
}




//------------------------------------ TENSOR STRUCT -------------------------------------------



/// A basic creator for the Tensor struct
/// Each Tensor is also given its own allocator
pub fn Tensor(comptime T: type) type {
    return struct {
        // -------------- members ---------------
        _allocator   : std.mem.Allocator,
        _data        : []T              ,
        _shape       : []usize          ,
        _strides     : [] usize         ,
        _owns_memory : bool = true      ,
        
        const Self = @This(); // so methods can directly use 'Self' instead of @This() everywhere
        
        // --------------------- constructors and destructors --------------------------

        /// The constructor for the tensor struct
        /// requires allocator and shape as input
        pub fn init(allocator : std.mem.Allocator, shape: []const usize) !Self{
            // because we dont own the memory the original shape points to:
            const shape_copy = try allocator.dupe(usize, shape); // duplicates
            // allocating the strides on the heap too
            const strides    = try allocator.alloc(usize, shape.len); 
            computeStrides(shape_copy, strides);
            const data = try allocator.alloc(T, numElements(shape));

            return Self{
                ._data      = data       ,
                ._allocator = allocator  , 
                ._shape     = shape_copy ,
                ._strides   = strides    ,
            };
        }
        
        /// Destructor for the tensor struct.
        /// Frees data (if owned by the Tensor)
        /// and the shape and strides array as well
        pub fn destroy(self: *Self) void{
            if (self._owns_memory) 
                self._allocator.free(self._data);
            self._allocator.free( self._shape );
            self._allocator.free(self._strides);
        }

        // ---------------------------- other methods -------------------------------------

        /// converts multi-dimensional indices into a single index for the flattened _data array
        fn offset(self : Self, indices: []const usize) usize {
            // EXPLAINATION :
            // user wants : [1, 2]
            // but we have flattened array stored
            // so it converts it into the index we want

            std.debug.assert(indices.len == self._shape.len);
            var off:usize = 0;
            for (indices, 0..) |idx, i| {
                off += idx * self._strides[i]; // standard row major calculation
            }
            return off;
        }
        /// returns the value of the index passed by the user
        /// internally calls the offset function to get the index in row-major format
        pub fn get(self: Self, indices: []const usize) *T{
            return &self._data[self.offset(indices)];
        }
    };
}



// ------------------------------------------- TESTS -----------------------------------------------



test "computeStrides" {
    // example :
    // [[1, 2, 3]
    //  [4, 5, 6]]
    // => in row major == [1, 2, 3, 4, 5, 6]
    // therefore stride = [3, 1] (jumping a row == move front/back thrice and jumping a col => moving once)
    const shape = [_]usize{ 2, 3 };
    var strides = [_]usize{ 0, 0 };

    computeStrides(shape[0..], strides[0..]);
    try std.testing.expectEqual(@as(usize, 3), strides[0]);
    try std.testing.expectEqual(@as(usize, 1), strides[1]);
}
