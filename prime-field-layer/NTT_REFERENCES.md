# NTT implementation references

The NTT and convolution implementation in this crate was independently written. No
third-party implementation source was copied or translated. In particular,
`concrete-ntt` was not used because its patent-license situation is outside this
project's scope.

The design was informed by the following published literature and public project
metadata:

- David Harvey, "Faster arithmetic for number-theoretic transforms," *Journal of
  Symbolic Computation* 60 (2014), DOI
  [10.1016/j.jsc.2013.09.002](https://doi.org/10.1016/j.jsc.2013.09.002).
  The lazy Shoup kernels implement the redundant-representation butterflies of
  this paper: the forward Cooley-Tukey butterfly follows section 4, Algorithm 4
  (residues in `[0, 4p)`, one conditional halving of the `X` input), and the
  inverse Gentleman-Sande butterfly follows section 3, Algorithm 3 (residues in
  `[0, 2p)`, one conditional halving of the sum, the `+2p`-planted difference
  fed unreduced into the multiplication). Both rely on the paper's observation
  that Shoup's multiplication needs no correction for any input below the word
  size, whose precise 32-bit-lane bound is restated and proved in the comments
  of `shoup_mul_lazy_for` in `src/ntt/support.rs`.
- Michael Scott, "A Note on the Implementation of the Number Theoretic
  Transform," (2017), DOI
  [10.1007/978-3-319-71045-7_13](https://doi.org/10.1007/978-3-319-71045-7_13).
- Patrick Longa and Michael Naehrig, "Speeding up the Number Theoretic Transform
  for Faster Ideal Lattice-Based Cryptography," (2016), DOI
  [10.1007/978-3-319-48965-0_8](https://doi.org/10.1007/978-3-319-48965-0_8).
- Gregor Seiler, "Faster AVX2 optimized NTT multiplication for Ring-LWE lattice
  cryptography," IACR ePrint [2018/039](https://eprint.iacr.org/2018/039).
- Jonathan Bradbury, Nir Drucker, and Marius Hillenbrand, "NTT software
  optimization using an extended Harvey butterfly," IACR ePrint
  [2021/1396](https://eprint.iacr.org/2021/1396). Its Theorem 2 analyzes the
  wide-input lazy Shoup product on 32-bit SIMD lanes with planted offsets in
  the multiplicand, which is the setting used by the Shoup kernels here; the
  bound is derived independently in the `shoup_mul_lazy_for` comments.
  The reduced Shoup kernel for `2^30 <= p < 2^31` uses the same quotient
  estimate with canonical inputs. Its uncorrected product is in `[0, 2p)`, then
  one correction keeps every butterfly input and output in `[0, p)`.
- Joris van der Hoeven and Grégoire Lecerf, HAL
  [hal-04841449](https://hal.science/hal-04841449).
- MIT-licensed FasterNTT source at commit
  [`95179634d2edbd7ae9e1ad3c39b8609bf74d20a6`](https://github.com/nict-sfl/FasterNTT/commit/95179634d2edbd7ae9e1ad3c39b8609bf74d20a6)
  was used only as a bibliographic comparison point; its source was not copied.
- Peter L. Montgomery, "Modular Multiplication Without Trial Division,"
  *Mathematics of Computation* 44.170 (1985), DOI
  [10.2307/2007970](https://doi.org/10.2307/2007970), underlies the wide-modulus
  fallback representation already used by this crate.
