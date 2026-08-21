In this project performance is very important.
Your code should be fast and use state-of-the-art algorithms.
This code will be used for cryptography purposes, so prefer constant-time whenever possible.

Benchmarks live in `prime-field-layer/benches`, split by purpose: `field` (field arithmetic), `dynamic_ntt` (auto-dispatch NTT plans), `static_ntt` (fixed-size plans), `backends` (forced scalar vs AVX2), `crossover` (schoolbook/NTT dispatch crossover), and `compare_static_dynamic_ntt` (static vs dynamic plans).

For development iterations run `cargo bench --features bench-quick`: it runs trimmed matrices of `field`, `dynamic_ntt`, and `static_ntt` only (individual functions, comparison targets are skipped) and takes a few minutes.

The full `cargo bench` suite preserves the historical benchmark ids, so criterion baselines stay comparable across the split, but it takes a while: set a long timeout (30 minutes should be enough).
