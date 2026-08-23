# Bibliography

- David Harvey, "Faster arithmetic for number-theoretic transforms,"
  *Journal of Symbolic Computation* 60 (2014), 113-119.
  [doi:10.1016/j.jsc.2013.09.002](https://doi.org/10.1016/j.jsc.2013.09.002).
  Used for the Shoup multiplication bounds and reduced and lazy NTT
  butterflies in `prime-field-layer`.
- Jonathan Bradbury, Nir Drucker, and Marius Hillenbrand, "NTT software
  optimization using an extended Harvey butterfly," IACR Transactions on
  Cryptographic Hardware and Embedded Systems 2022.4 (2022), 311-334.
  [doi:10.46586/tches.v2022.i4.311-334](https://doi.org/10.46586/tches.v2022.i4.311-334).
  Used for the 32-bit SIMD Shoup-product analysis.
- Peter L. Montgomery, "Modular Multiplication Without Trial Division,"
  *Mathematics of Computation* 44.170 (1985), 519-521.
  [doi:10.2307/2007970](https://doi.org/10.2307/2007970).
  Used for field-element pointwise multiplication and inverse normalization.
- Michael O. Rabin, "Probabilistic Algorithms in Finite Fields," *SIAM Journal
  on Computing* 9.2 (1980), 273-280.
  [doi:10.1137/0209024](https://doi.org/10.1137/0209024).
  Used for the deterministic Frobenius and polynomial-GCD irreducibility
  criterion in `prime-field-layer::extension_field`.
- Andrew V. Sutherland, "Finite field arithmetic," MIT 18.783 lecture notes 4
  (2019), sections 4.1-4.2.
  [Lecture notes](https://math.mit.edu/classes/18.783/2019/LectureNotes4.pdf).
  Used for polynomial reversal, truncated inverse, and fast Euclidean division
  in fixed monic reduction.
- Daniel Lemire, "Fast Random Integer Generation in an Interval," *ACM
  Transactions on Modeling and Computer Simulation* 29.1 (2019), article 3.
  [doi:10.1145/3230636](https://doi.org/10.1145/3230636).
  Used for the unbiased reduction of fixed-width random words to a bounded
  integer by rejecting the incomplete range.
- Fabrice Benhamouda, Caicai Chen, Shai Halevi, Yuval Ishai, Hugo Krawczyk,
  Tamer Mour, Tal Rabin, and Alon Rosen, "Encrypted Matrix-Vector Products from
  Secret Dual Codes," IACR ePrint 2025/858 (2025), Definition 2.3 and Sections
  5.1-5.2. [ePrint](https://eprint.iacr.org/2025/858).
  Primary specification for all three constructions in `trapdoor-matrices`.
- Elette Boyle, Geoffroy Couteau, Niv Gilboa, Yuval Ishai, Lisa Kohl, and Peter
  Scholl, "Efficient Pseudorandom Correlation Generators from Ring-LPN," IACR
  ePrint 2022/1035; CRYPTO 2020. [ePrint](https://eprint.iacr.org/2022/1035).
  Used for the irreducible-extension Ring-LPN assumption and its structural
  security caveats.
- Stefan Heyse, Eike Kiltz, Vadim Lyubashevsky, Christof Paar, and Krzysztof
  Pietrzak, "Lapin: An Efficient Authentication Protocol Based on Ring-LPN,"
  FSE 2012, LNCS 7549, 346-365.
  [preprint](https://www.iacr.org/workshops/fse2012/FSEpreproceedings/PDF/19.pdf).
  Used for the conservative choice of an irreducible Ring-LPN modulus.
- Martijn Brehm, Binyi Chen, Ben Fisch, Nicolas Resch, Ron D. Rothblum, and
  Hadas Zeilberger, "Blaze: Fast SNARKs from Interleaved RAA Codes," IACR
  ePrint 2024/1609; EUROCRYPT 2025.
  [ePrint](https://eprint.iacr.org/2024/1609).
  Used only as algorithmic background for repeat-accumulate-accumulate codes;
  its coding results do not prove pseudorandomness of the weighted TDM.
- Guy E. Blelloch, "Prefix Sums and Their Applications," in *Synthesis of
  Parallel Algorithms* (1993).
  [author-hosted paper](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf).
  Used for the weighted-prefix-scan formulation of the RAA evaluator.

## Dependency notes

- `rand_core` 0.9.5 supplies the maintained `RngCore` and `CryptoRng` traits for
  caller-owned cryptographic RNGs. It is used without default features, so the
  library gains no OS RNG or implicit seeding. The crate is dual-licensed under
  MIT or Apache-2.0.
  [crate documentation](https://docs.rs/rand_core/0.9.5/rand_core/).
- `rand_chacha` 0.9.0 is a development-only dependency used for reproducible
  sampling tests. It is dual-licensed under MIT or Apache-2.0 and does not form
  part of the library API.
  [crate documentation](https://docs.rs/rand_chacha/0.9.0/rand_chacha/).
- `allocation-counter` 0.8 is a development-only dependency used to prevent
  allocation regressions in warmed TDM evaluation. It is dual-licensed under
  MIT or Apache-2.0 and installs its counting allocator only in test binaries.
  [crate documentation](https://docs.rs/allocation-counter/0.8/allocation_counter/).
