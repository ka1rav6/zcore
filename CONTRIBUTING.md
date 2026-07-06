# Contributing to zcore

Thank you for your interest in contributing to zcore!

## Getting Started

1. Fork the repository.
2. Clone your fork:
   ```sh
   git clone https://github.com/your-username/zcore.git
   ```
3. Create a feature branch:
   ```sh
   git checkout -b feat/my-feature
   ```

## Development

Build the project:
```sh
zig build
```

Run tests:
```sh
zig build test
```

Run the executable:
```sh
zig build run
```

## Code Style

- **Zig version:** Match the version of Zig used throughout the project.
- **No AI-generated code:** Per Zig community norms, do not use AI tools for any purpose — including code, documentation, tests, or debugging.
- Follow the [Zig Style Guide](https://ziglang.org/documentation/master/#Style-Guide).
- Use 4-space indentation.
- Use PascalCase for struct/enum/union names and `snake_case` for function/variable names. Prefix struct members with an underscore (e.g., `_strides: []usize`).
- Keep functions focused and small.
- **Every function must have tests.** If tests are not yet written, open an issue stating that the feature lacks test coverage.
- **Write thorough, beginner-friendly comments.** The source code should serve as a learning resource for readers of all levels.
- Document and explain every design decision with inline comments.
- Re-export all public API from `src/root.zig`.

## Pull Requests

- Keep PRs focused on a single concern.
- Write clear commit messages.
- Ensure all tests pass before submitting.
- Update documentation if your change introduces new behavior.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
