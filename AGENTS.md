# EMVP Experiments

This project contains various experiments about the Encrypted Matrix-Vector Product algorithm described in `../paper/2025-858.pdf` and is part of the work for my thesis.

## Code style

Performance is very important.
Your code should be fast and use state-of-the-art algorithms.
This code will be used for cryptography purposes, so prefer constant-time whenever possible.

## Benchmarks

Benchmarks live in `prime-field-layer/benches`, split by purpose: 
- `field` (field arithmetic)
- `dynamic_ntt` (auto-dispatch NTT plans)
- `static_ntt` (fixed-size plans)
- `backends` (forced scalar vs AVX2)
- `crossover` (schoolbook/NTT dispatch crossover)
- `compare_static_dynamic_ntt` (static vs dynamic plans)

Always benchmark your changes. If no suitable benchmark exist write a new one.

For development iterations run `cargo bench --features bench-quick`: it runs trimmed matrices of `field`, `dynamic_ntt`, and `static_ntt` only and only takes a few minutes.

The full `cargo bench` suite preserves the historical benchmark ids, so criterion baselines stay comparable across the split, but it takes a while: set a long timeout (60 minutes should be enough).

## References

Always cite your references, so that I can use them in my bibliography. Keep track of them in `BIBLIOGRAPHY.md`.
