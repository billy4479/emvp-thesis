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
  [2021/1396](https://eprint.iacr.org/2021/1396).
- Joris van der Hoeven and Grégoire Lecerf, HAL
  [hal-04841449](https://hal.science/hal-04841449).
- MIT-licensed FasterNTT source at commit
  [`95179634d2edbd7ae9e1ad3c39b8609bf74d20a6`](https://github.com/nict-sfl/FasterNTT/commit/95179634d2edbd7ae9e1ad3c39b8609bf74d20a6)
  was used only as a bibliographic comparison point; its source was not copied.
- Peter L. Montgomery, "Modular Multiplication Without Trial Division,"
  *Mathematics of Computation* 44.170 (1985), DOI
  [10.2307/2007970](https://doi.org/10.2307/2007970), underlies the wide-modulus
  fallback representation already used by this crate.
