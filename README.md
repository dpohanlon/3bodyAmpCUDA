# 3bodyAmpCUDA

Prototype 3-body decay amplitude computation in CUDA, for [Laura++](https://laura.hepforge.org/).

Branch divergence results in a significant performance penalty on GPUs. This is a prototype implementation of how static dispatch using C++ template specialisation (rather than function pointers, or virtual functions) can be used to remove run-time branches in CUDA kernels.

Additionally, as all of the potential branches are known and evaluated at compile time, this allows the compiler to optimise through the function calls, resulting in more efficient generated instructions than other methods.

Ultimately, this method could be used to fuse the amplitude calculation into a single CUDA compute kernel.
