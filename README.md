# zcore

**A tensor library written in Zig, for Zig.**

zcore aims to be Zig's answer to NumPy/PyTorch - a fast, well-documented tensor library that's as educational as it is practical. Every line is written with beginners in mind. It is first in the line of libraries for zig: a dataframe (pandas equivalent) and a plotting library. Zig is great for such a library and it could be the future of ML engines/ AI engines if such libraries exist

```zig
const zcore = @import("zcore");

const allocator = std.heap.page_allocator;
var t = try zcore.Tensor(f32).zeroes(allocator, &[_]usize{ 2, 3 });
defer t.destroy();

t.fill(1.0);
t.get(&[_]usize{ 0, 1 }).* = 42.0;
```

## Why contribute?

- **Low barrier to entry right now** -> the codebase is small, modular, and heavily commented. You don't need to be a linear algebra expert to make a meaningful PR.
- **Clear roadmap** -> there's a detailed [implementation plan](implementation_plan.md) with bite-sized tasks: reshape, transpose, broadcasting, element-wise ops, and more.
- **Learn by doing** -> contribute to a real library while deepening your understanding of Zig, memory management, SIMD, and numerical computing.
- **Zero deps** -> no build system headaches, no bloated dependencies. Just `zig build test`.
- **Your kind of people** -> no AI-generated code. Just humans writing thoughtful, commented Zig.

## Quick start

```sh
git clone https://github.com/ka1rav6/zcore
cd zcore
chmod +x ./run_tests.sh
./run_tests # run the tests
zig build example1 # run the first example
```

Requires Zig **0.15.2+**.

## Current state

|           Feature          |    Status         |
|----------------------------|-------------------|
| Generic `Tensor(T)`        | majorly completed |
| Shape / strides            | majorly completed |
| row-major storage          | majorly completed |
| `init`, `fill`, `zeroes`   |      completed    |
| Multi-dimensional indexing (`get`) | completed |
| Memory ownership (views)   | majorly completed |
| Reshape, transpose, slice  | majorly completed |
| Element-wise arithmetic    | Soon |
| Broadcasting               | Soon |
| Matrix arithmetics         | Soon |
| Full NumPy-level API       | Soon |

## How to contribute

1. Read [CONTRIBUTING.md](CONTRIBUTING.md).
2. Pick something (a logical next step) from the [implementation plan](implementation_plan.md) or read the issues for something.
3. Read [STYLE_GUIDE.md](STYLE_GUIDE.md).
4. Open a PR.

Every function must have tests. Every design decision must be explained in comments. Contributions of all sizes are welcome — whether it's a bug fix, a new feature, or just better documentation.

## License

MIT
