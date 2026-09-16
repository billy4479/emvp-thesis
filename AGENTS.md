# EMVP Experiments

This folder is a workspace for my thesis work, which is based on the EMVP paper which you can find in `../paper/2025-858.pdf`.
My work is about implementing the protocol suggested in the paper in particular for LLM-scale matrix operations.

## Code style

- Performance is very important.
- Your code should be fast and use state-of-the-art algorithms.
- This code will be used for cryptography purposes, so prefer constant-time whenever possible.
- Simple but slightly slower is better than complex but slightly faster.

## Tests

Correctness comes before anything else. Write useful tests for your changes, which actually make sure the code is doing the right thing.

## Benchmarks 

When you need to implement a new feature or change something you should follow this procedure:
- Look at the preexisting benchmarks and assess if they already cover what you are going to change.
- If they are insufficient, write new ones.
- Pick a subsection of benchmarks to run (the whole suite is very long to run) and save a baseline.
- Implement your change.
- Measure again against the baseline.

## References

Always cite your references, so that I can use them in my bibliography.
Keep track of them in `BIBLIOGRAPHY.yml`.
This file uses the Hayagriva bibliography format, more information at `https://raw.githubusercontent.com/typst/hayagriva/refs/heads/main/docs/file-format.md`
