# Agent Instructions

## Code style

- Do not write comments or docstrings. Make code explain itself with clearly named functions and variables instead.
- Name functions after what they do (`normalize_observation`, `_next_generation`) and variables after what they hold (`survivor_count`, `consecutive_hits`). Avoid vague names like `x`, `tmp`, `data`, or `result2`.
- When a block of logic needs explaining, pull it into a well-named function rather than adding a comment.
- Use named constants for magic numbers (`ROUND_WIN_REWARD = 100.0`, not a bare `100`).
