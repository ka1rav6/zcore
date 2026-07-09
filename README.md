# zcore

**A tensor library written in Zig, for Zig.**

zcore aims to be Zig's answer to NumPy/PyTorch — a fast, well-documented tensor library that's as educational as it is practical. Every line is written with beginners in mind. It is the first in a planned line of data-science libraries for Zig: a dataframe (pandas equivalent) and a plotting library.

```zig
const zcore = @import("zcore");

const allocator = std.heap.page_allocator;
var t = try zcore.Tensor(f32).zeroes(allocator, &[_]usize{ 2, 3 });
defer t.destroy();

t.fill(1.0);
t.get(&[_]usize{ 0, 1 }).* = 42.0;
```

## Installation

Add `zcore` as a dependency in `build.zig.zon`:

```sh
zig fetch --save https://github.com/ka1rav6/zcore/archive/refs/tags/v0.1.0.tar.gz
```

Then import the module in your `build.zig`:

```zig
const zcore = b.dependency("zcore", .{
    .target = target,
    .optimize = optimize,
});
exe.root_module.addImport("zcore", zcore.module("zcore"));
```

## Quick start

```sh
git clone https://github.com/ka1rav6/zcore
cd zcore
zig build test        # run the test suite
zig build example1    # run the first example
```

Requires Zig **0.17.0-dev.1158+1d1193aa7** (see CI for exact pinned version).

## Why contribute?

- **Low barrier to entry** — the codebase is small, modular, and heavily commented. You don't need to be a linear algebra expert to make a meaningful PR.
- **Clear roadmap** — there's a detailed [implementation plan](implementation_plan.md) with bite-sized tasks: reshape, transpose, broadcasting, element-wise ops, and more.
- **Learn by doing** — contribute to a real library while deepening your understanding of Zig, memory management, SIMD, and numerical computing.
- **Zero deps** — no build system headaches, no bloated dependencies. Just `zig build test`.
- **Your kind of people** — no AI-generated code. Just humans writing thoughtful, commented Zig.

## Current state (v0.1.0)

| Feature                    | Status  |
|----------------------------|---------|
| Generic `Tensor(T)`        | Done    |
| Shape / strides            | Done    |
| Row-major storage          | Done    |
| `init`, `fill`, `zeroes`   | Done    |
| Multi-dimensional indexing | Done    |
| Memory ownership (views)   | Done    |
| Transpose, slice, resize   | Done    |
| Broadcasting               | Done    |
| 0-length tensor support    | Done    |
| Element-wise arithmetic    | Planned |
| Matrix arithmetic          | Planned |
| Full NumPy-level API       | Planned |

See the [implementation plan](implementation_plan.md) for the full roadmap.

## Documentation

- [Implementation Plan](implementation_plan.md) — phased roadmap from v0.1.0 through v1.0
- [Style Guide](STYLE_GUIDE.md) — code conventions
- [Contributing Guide](CONTRIBUTING.md) — how to get involved
- [Changelog](CHANGELOG.md) — release history

## How to run

| Command              | Description           |
|----------------------|-----------------------|
| `zig build test`     | Run all tests         |
| `zig build run`      | Run the executable    |
| `zig build example1` | Run the first example |

## License

MIT — see [LICENSE](LICENSE).
