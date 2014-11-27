70 lines of NumPy
=================

This code solves the MuMag standard problem #4 in less than 70 lines of pure
Python without counting empty lines and comments :). The code makes largely
use of the NumPy library and computes the exchange field by finite differences
and the demagnetization field with a fast convolution algorithm. Since the
magnetization in finite-difference micromagnetics is represented by a
multi-dimensional array and the NumPy library features a rich interface for
this data structure, the presented code is a good starting point for the
development of novel algorithms.

A detailed description of the code can be found here: http://arxiv.org/abs/1411.7188
