# Changelog

All notable changes to zcore are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] -> 2026-07-09

### Added

- **Generic `Tensor(T)`** -> type-safe tensor over `u8`, `u32`, `u64`, `i32`, `i64`, `f32`, `f64`, and `bool`. Comptime gate rejects non-numeric types.
- **Shape / strides system** -> row-major (C-style) storage with automatic stride computation.
- **Constructors** -> `init`, `zeroes`, `ones`, `empty`, `full`, `from_slice`.
- **Indexing** -> `get` (bounds-checked pointer), `at` (bounds-checked value), `get_unchecked` (unsafe), `set` (bounds-checked write).
- **Row/column operations** -> `set_row`, `set_col`, `set_whole`.
- **Fill** -> `fill` sets every element to a given value.
- **Shape manipulation** -> `resize`, `transpose` (2D, O(1) stride-swap), `slice` (first-dimension view).
- **NumPy-style broadcasting** -> `broadcastShape` (shape inference), `broadcast` (two-tensor view), `broadcast_to` (single-tensor view). Uses stride=0 for broadcast dimensions, no data copy.
- **Debug printing** -> `debug_print` recursively formats tensors of any dimensionality, handles scalars (0-d) and 0-length tensors.
- **Memory ownership** -> views (`slice`, `broadcast_to`) set `_owns_memory = false`; `destroy` only frees owned data.
- **0-length tensor support** -> all operations correctly handle shapes containing zero dimensions (e.g. `[0]`, `[3, 0]`, `[0, 3]`). Indexing any 0-length dimension returns `IndexOutOfBounds`.
- **Comprehensive test suite** -> 30+ unit tests covering constructors, indexing, slicing, transposing, resizing, broadcasting, edge cases (0-d scalars, 0-length tensors), and comptime type rejection.
- **Build system** -> `build.zig` + `build.zig.zon` exposing `zcore` as a Zig module. `zig build test` runs all tests.
- **CI pipeline** -> GitHub Actions: format check (`zig fmt`), test (`zig build test`), and build (`zig build`) on every push/PR.
- **Examples** -> `examples/simple1.zig` and standalone integration test `tests/test1.zig`.
- **Documentation** -> `CONTRIBUTING.md`, `STYLE_GUIDE.md`, `implementation_plan.md` with phased roadmap.

### Fixed

- `at` method returned a pointer (`*T`) instead of a value (`T`). Now correctly returns the element by value.

[0.1.0]: https://github.com/ka1rav6/zcore/releases/tag/v0.1.0
