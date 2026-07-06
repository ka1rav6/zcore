const std = @import("std");
const utils = @import("utils.zig");

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
            utils.computeStrides(shape_copy, strides);
            const data = try allocator.alloc(T, utils.numElements(shape));

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

        /// fills the whole tensor with the value taken in input
        /// takes O(n) time where n is the size of the data
        pub fn fill(self: *Self, value: T) void{
            for (self._data) |*x|{
                x.* = value;
            }
        }

        /// Initializes the tensor and returns the instance of the tensor
        /// with zeroes filled inside the data by default
        pub fn zeroes(allocator : std.mem.Allocator, shape: []const usize) !Self {
            var temp_t:Self = undefined;
            temp_t = try Self.init(allocator, shape); // calling the constructor
            @memset(temp_t._data, 0); 
            return temp_t;
        }
    };
}
