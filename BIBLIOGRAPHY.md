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
