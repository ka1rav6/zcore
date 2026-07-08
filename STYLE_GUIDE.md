# Style Guide

## Naming 

- Functions          : snake_case
- Function params    : snake_case
- Types              : PascalCase
- Struct members     : _whatever_this_case_is
- Constants          : SCREAMING_SNAKE_CASE
- File names         : snake_case
- Errors/Error names : PascalCase

## Comments

Add brutally verbose and clear comments.
Even beginners should be able to understand each and every part of the code.
We want good contributions. 
If people understand it, people will use it, and people will contribute to it
Add documentation comments to every single function, whether private or public

## Errors

Always return errors instead of formatting

## Formatting

Github actions automatically use `zig fmt`. 
Still, it is preferable to format it and follow the conventions to maintain consistency
Follow the 3-space indentation
