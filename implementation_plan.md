# Zcore implementation plan:

## First Learn:
--- zig related ---
1. zig completely
2. Memory allocators/alignement
3. Pointer Arithmetic
4. Slices
5. Cache locality
 ---
6. Cache lines
7. Branch Prediction
8. SIMD
9. Vector Instructions

---
Linear Algebra:

10. Vectors
11. Matrices
12. Tensor Dimensions
13. Matrix Multiplication
14. Transposing
15. Dot Products
16. Broadcasting
---
Study:

17. Numpy Internals
18. Pytorch tensor part
19. How a tensor is actually implemented


## Actual Implementation Phase:
_what each version should have_

### Version 0.1
- Tensor Representation
- Shape
- Stride
- Memory Management (tensor storage etc)
- Supporting different numerical types (f32, f64, i32, i64, u8, bool, i64...)
- Define NaN?
- Tensor Constructions : copying, moving, empty(), full(), zeroes() etc
- Tensor Indexing

### Version 0.5 :
- Reshape
- Transpose
- Slice
- Flatten
- Permute
- Expand Dimensions
- Squeeze
- Iterator Creation
- Tensor Arithmetic : + - * / % pow sqrt abs exp log cos sin tanh ...
- Broadcasting rules
- Comparison ( + Logical ) 
- Sum, mean, min, maz, argmax, argmin, variance, standard deviation, product

## Version 1.1

- Matrix Multiplication :
    - first normal O(n<sup>3</sup>)
    - Then O(n<sup>2.81</sup>)
    - Then SIMD
    - Multithreaded possibilities
    - BLAS backend ?

- Linear Algebra :
    - Dot Product
    - Cross Product
    - Inversing
    - LU transformation
    - QR transformation
    - SVD
    - Cholesky
    - Eigen Decomposition
    - Determinants
    - Norms
### Version 1.5

- Uniform Distribution
- Normal Distribution
- Bernoulli Distribution
- Randint?
- Shuffle
- Permutation
- Seeding

### Version 2.0

 [[ TO BE CONTINUED ]]


---








## Design Choices to implement

1. NaN -> defining on my own
2. +-0, += inf
3. DType : implementing myself

```zig
// something like this: 
const DType = enum { f32, f64, i32, i64, bool };
```

Follow IEEE as much as possible






