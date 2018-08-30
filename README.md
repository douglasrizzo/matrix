# C++ matrix implementation

This is an implementation of the mathematical matrix I made to use in other projects of mine, mostly related to [statistical and machine learning](http://douglasrizzo.com.br/machine_learning).

Internally, it stores arithmetic values in a C++ vector and uses OpenMP collapsed parallel for loops to do things faster, if possible.

Please note that I am in no way an expert in C++ or advanced linear algebra, so things might not be done as mathematically or memory efficient as possible. Still, some benchmarks were made to see the best conditions on when to trigger OpenMP and some optimizations on memory access were also implemented.

The class contains the following features:

 - operator overloading for scalar and matrix addition, subtraction, multiplication as well as matrix equality and inequality
 - cofactor and cofactor matrix calculation
 - Hadamard (entrywise) multiplication
 - adjugate, determinant and inverse
 - transpose
 - adding and removing columns and rows
 - get a single column or row
 - get a subset of rows or columns
 - create submatrices (a new matrix from a preexisting one, removing a single column and row)
 - count occurrences, sort elements in-place, return only unique values from a matrix
 - mean, scatter, variance, covariance and standard deviation from a matrix or each of its columns
 - reshape
 - standardize columns by the z-score (subtract mean and divide by stddev)
 - create a diagonal matrix from column vector
 - normalize columns (divide each column vector by the length of the vector)
 - calculate eigenvalues and eigenvectors of symmetric and non-symmetric matrices (uses algorithm from Numerical Recipes)
 - within-class and between-class scatter
 - one-hot encoding of the unique values of a matrix