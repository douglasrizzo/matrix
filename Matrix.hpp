/**
 * @author Douglas De Rizzo Meneghetti (douglasrizzom@gmail.com)
 * @brief  Matrix implementation, with a series of linear algebra functions
 * @date   2017-10-17
 */

#ifndef MACHINE_LEARNING_MATRIX_HPP
#define MACHINE_LEARNING_MATRIX_HPP

#include <vector>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <set>
#include <complex>
#include "include/nr3/nr3.h"
#include "include/nr3/eigen_unsym.h"
#include "include/csv_reader/CSVReader.hpp"

using namespace std;

//! Matrix implementation, with a series of linear algebra functions
//! @tparam T The arithmetic type the matrix will store
template<
    typename T,
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
class Matrix {
 private:
  size_t mRows;
  size_t mCols;
  std::vector<T> mData;

  //! Validates if indices are contained inside the matrix
  //! \param row row index
  //! \param col column index
  //! \throws runtime error if at least one of the indices is out of bounds
  void validateIndexes(size_t row, size_t col) const {
    if (row < 0 or row >= mRows)
      throw invalid_argument(
          "Invalid row index (" + to_string(row) + "): should be between 0 and " + to_string(mRows - 1));
    if (col < 0 or col >= mCols)
      throw invalid_argument(
          "Invalid column index (" + to_string(col) + "): should be between 0 and " + to_string(mCols - 1));
  }

  //! Sorts eigenvalues by magnitude, sorting their corresponding eigenvectors in te same order
  //! \param eigenvalues column vector containing eigenvalues
  //! \param eigenvectors Matrix containing the eigenvectors in its columns.
  //! It must have the same number of columns as <code>eigenvalues</code> has rows
  //! \return
  static pair<Matrix, Matrix> eigsort(Matrix eigenvalues, Matrix eigenvectors) {
//    if (eigenvalues.mCols != eigenvectors.mRows)
//      throw runtime_error("Incompatible number of eigenvalues and eigenvectors");

    Matrix eigval(eigenvalues.mRows, eigenvalues.mCols, eigenvalues.mData);
    Matrix eigvec(eigenvectors.mRows, eigenvectors.mCols, eigenvectors.mData);

    // keep the order of eigenvalues in this vector
    vector<size_t> newOrder;
    for (size_t i = 0; i < eigenvalues.nRows(); i++) {
      int position = 0;
      for (int j = 0; j < newOrder.size(); j++)
        if (eigenvalues(i, 0) < eigenvalues(newOrder[j], 0))
          position++;
      newOrder.insert(newOrder.begin() + position, i);
    }

    // order eigenvalues and eigenvectors by the value of the eigenvalues
    for (size_t i = 0; i < newOrder.size(); i++) {
      eigval(i, 0) = eigenvalues(newOrder[i], 0);

      for (int j = 0; j < eigenvectors.nRows(); j++) {
        eigvec(static_cast<size_t>(j), i) = eigenvectors(j, newOrder[i]);
      }
    }

    return make_pair(eigval, eigvec);
  }

 public:

  enum Axis { ALL, ROWS, COLUMNS };

  size_t nCols() const { return mCols; }

  size_t nRows() const { return mRows; }

  //region Constructors

  //! Initializes an empty matrix
  Matrix() {
    mRows = mCols = 0;
  }

  //! Initializes a square matrix
  //! \param dimension number of rows and columns
  Matrix(size_t dimension) {
    Matrix(dimension, dimension);
  }

  //! Initializes a matrix with a predetermined number of rows and columns
  //! \param rows number of rows in the matrix
  //! \param cols number of columns in the matrix
  Matrix(size_t rows, size_t cols)
      : mRows(rows),
        mCols(cols),
        mData(rows * cols) {
  }

  //! Initializes a matrix with a predetermined number of rows and columns and populates it with data
  //! \param rows number of rows in the matrix
  //! \param cols number of columns in the matrix
  //! \param data a vector containing <code>rows * cols</code> elements to populate the matrix
  Matrix(size_t rows, size_t cols, const vector<T> &data)
      : mRows(rows),
        mCols(cols) {
    if (data.size() != rows * cols)
      throw invalid_argument("Matrix dimension incompatible with its initializing vector.");
    mData = data;
  }

  template<std::size_t N>
  Matrix(size_t rows, size_t cols, T (&data)[N]) {
    if (N != rows * cols)
      throw invalid_argument("Matrix dimension incompatible with its initializing vector.");
    vector<T> v(data, data + N);
    Matrix(rows, cols, v);
  }
  //endregion

  //region Operators

  //region Scalar operators

  //! Scalar addition
  //! \param m a matrix
  //! \param value scalar to be added to the matrix
  //! \return the result of the scalar addition of <code>m</code> and <code>value</code>
  friend Matrix operator+(const Matrix &m, double value) {
    Matrix result(m.mRows, m.mCols);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m.mRows; i++) {
      for (size_t j = 0; j < m.mCols; j++) {
        result(i, j) = value + m(i, j);
      }
    }

    return result;
  }

  //! Scalar addition
  //! \param m a matrix
  //! \param value scalar to be added to the matrix
  //! \return the result of the scalar addition of <code>m</code> and <code>value</code>
  friend Matrix operator+(double value, const Matrix &m) {
    return m + value;
  }

  //! Scalar subtraction
  //! \param m a matrix
  //! \param value scalar to be subtracted to the matrix
  //! \return the result of the scalar subtraction of <code>m</code> and <code>value</code>
  friend Matrix operator-(const Matrix &m, double value) {
    return m + (-value);
  }

  //! Scalar subtraction
  //! \param m a matrix
  //! \param value scalar to be subtracted to the matrix
  //! \return the result of the scalar subtraction of <code>m</code> and <code>value</code>
  friend Matrix operator-(double value, const Matrix &m) {
    return m - value;
  }

  //! Scalar multiplication
  //! \param m a matrix
  //! \param value scalar to be multiplied by the matrix
  //! \return the result of the scalar multiplication of <code>m</code> and <code>value</code>
  friend Matrix operator*(const Matrix &m, double value) {
    Matrix result(m.mRows, m.mCols);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m.mRows; i++) {
      for (size_t j = 0; j < m.mCols; j++) {
        result(i, j) = value * m(i, j);
      }
    }

    return result;
  }

  //! Scalar multiplication
  //! \param m a matrix
  //! \param value scalar to be multiplied by the matrix
  //! \return the result of the scalar multiplication of <code>m</code> and <code>value</code>
  friend Matrix operator*(double value, const Matrix &m) {
    return m * value;
  }

  //! Scalar division
  //! \param m a matrix
  //! \param value scalar to be divide the matrix by
  //! \return the result of the scalar division of <code>m</code> by <code>value</code>
  friend Matrix operator/(const Matrix &m, double value) {
    Matrix result(m.mRows, m.mCols);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m.mRows; i++) {
      for (size_t j = 0; j < m.mCols; j++) {
        result(i, j) = m(i, j) / value;
      }
    }

    return result;
  }

  //! Scalar division
  //! \param value scalar that will be divided by the matrix
  //! \param m a matrix
  //! \return the result of the scalar division of <code>value</code> by <code>m</code>
  friend Matrix operator/(double value, const Matrix &m) {
    // division is not commutative, so a new method is implemented
    Matrix result(m.mRows, m.mCols);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m.mRows; i++) {
      for (size_t j = 0; j < m.mCols; j++) {
        result(i, j) = value / m(i, j);
      }
    }

    return result;
  }

  Matrix operator+=(double value) {
    #pragma omp parallel for
    for (int i = 0; i < mData.size(); i++)
      mData[i] += value;
    return *this;
  }

  Matrix operator-=(double value) {
    #pragma omp parallel for
    for (int i = 0; i < mData.size(); i++)
      mData[i] -= value;
    return *this;
  }

  Matrix operator*=(double value) {
    #pragma omp parallel for
    for (int i = 0; i < mData.size(); i++)
      mData[i] *= value;
    return *this;
  }

  Matrix operator/=(double value) {
    #pragma omp parallel for
    for (int i = 0; i < mData.size(); i++)
      mData[i] /= value;
    return *this;
  }
  //endregion

  //region Matrix operators

  //! Matrix addition operation
  //! \param b another matrix
  //! \return Result of the addition of both matrices
  Matrix operator+(const Matrix &b) {
    if (mRows != b.mRows || mCols != b.mCols)
      throw invalid_argument("Cannot add these matrices: L = " + to_string(mRows) + "x" + to_string(mCols) + ", R = "
                                 + to_string(b.mRows) + "x" + to_string(b.mCols));

    Matrix result(mRows, mCols);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < mRows; i++) {
      for (size_t j = 0; j < mCols; j++) {
        result(i, j) = operator()(i, j) + b(i, j);
      }
    }

    return result;
  }

  //! Matrix subtraction operation
  //! \param b another matrix
  //! \return Result of the subtraction of both matrices
  Matrix operator-(const Matrix &b) {
    if (mRows != b.mRows || mCols != b.mCols)
      throw invalid_argument(
          "Cannot subtract these matrices: L = " + to_string(mRows) + "x" + to_string(mCols) + ", R = "
              + to_string(b.mRows) + "x" + to_string(b.mCols));

    Matrix result(mRows, mCols);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < mRows; i++) {
      for (size_t j = 0; j < mCols; j++) {
        result(i, j) = operator()(i, j) - b(i, j);
      }
    }

    return result;
  }

  //! Matrix multiplication operation
  //! \param b another matrix
  //! \return Result of the multiplication of both matrices
  Matrix operator*(const Matrix &b) const {
    if (mCols != b.mRows)
      throw invalid_argument(
          "Cannot multiply these matrices: L = " + to_string(this->mRows) + "x" +
              to_string(this->mCols) + ", R = " + to_string(b.mRows) + "x" + to_string(b.mCols));

    Matrix result = zeros(mRows, b.mCols);

    #pragma omp parallel for if(result.mRows * result.mCols > 250)
    for (size_t i = 0; i < result.mRows; i++) {
      for (size_t k = 0; k < mCols; k++) {
        double tmp = operator()(i, k);
        for (size_t j = 0; j < result.mCols; j++) {
          result(i, j) += tmp * b(k, j);
        }
      }
    }

    return result;
  }

  Matrix &operator+=(const Matrix &other) {
    if (mRows != other.mRows || mCols != other.mCols)
      throw invalid_argument("Cannot add these matrices: L = " + to_string(mRows) + "x" + to_string(mCols) + ", R = "
                                 + to_string(other.mRows) + "x" + to_string(other.mCols));
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < other.mRows; i++) {
      for (size_t j = 0; j < other.mCols; j++) {
        operator()(i, j) += other(i, j);
      }
    }

    return *this;
  }

  Matrix &operator-=(const Matrix &other) {
    if (mRows != other.mRows || mCols != other.mCols)
      throw invalid_argument(
          "Cannot subtract these matrices: L = " + to_string(mRows) + "x" + to_string(mCols) + ", R = "
              + to_string(other.mRows) + "x" + to_string(other.mCols));

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < other.mRows; i++) {
      for (size_t j = 0; j < other.mCols; j++) {
        operator()(i, j) -= other(i, j);
      }
    }

    return *this;
  }

  Matrix &operator*=(const Matrix &other) {
    if (mCols != other.mRows)
      throw invalid_argument(
          "Cannot multiply these matrices: L " + to_string(mRows) + "x" +
              to_string(mCols) + ", R " + to_string(other.mRows) + "x" + to_string(other.mCols));

    Matrix result(mRows, other.mCols);

    #pragma omp parallel for collapse(2)
    // two loops iterate through every cell of the new matrix
    for (size_t i = 0; i < result.mRows; i++) {
      for (size_t j = 0; j < result.mCols; j++) {
        // here we calculate the value of a single cell in our new matrix
        result(i, j) = 0;
        for (size_t ii = 0; ii < mCols; ii++)
          result(i, j) += operator()(i, ii) * other(ii, j);
      }
    }

    mRows = result.mRows;
    mCols = result.mCols;
    mData = result.mData;
    return *this;
  }
  //endregion

  //region Equality operators

  Matrix<int> operator==(const T &value) {
    Matrix<int> result(mRows, mCols);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < mRows; i++) {
      for (size_t j = 0; j < mCols; j++) {
        result(i, j) = operator()(i, j) == value;
      }
    }

    return result;
  }

  bool operator==(const Matrix &other) {
    if (mData.size() != other.mData.size() || mRows != other.mRows || mCols != other.mCols)
      return false;

    for (int k = 0; k < mData.size(); k++) {
      if (mData[k] != other.mData[k])return false;
    }

    return true;
  }

  Matrix operator!=(const double &value) {
    // subtract 1 from everything: 0s become -1s, 1s become 0s
    // negate everything: 0s remains 0s, -1s becomes 1s
    return -((*this == value) - 1);
  }

  bool operator!=(const Matrix &other) {
    // subtract 1 from everything: 0s become -1s, 1s become 0s
    // negate everything: 0s remains 0s, -1s becomes 1s
    return !(*this == other);
  }
  //endregion

  //! Matrix negative operation
  //! \return The negative of the current matrix
  Matrix operator-() {
    Matrix result(this->mRows, this->mCols);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < mCols; i++) {
      for (size_t j = 0; j < mRows; j++) {
        result(i, j) = -operator()(i, j);
      }
    }

    return result;
  }

  //region Functors

  //! Functor used to access elements in the matrix
  //! \param i row index
  //! \param j column index
  //! \return element in position ij of the matrix
  T &operator()(size_t i, size_t j) {
    validateIndexes(i, j);
    return mData[i * mCols + j];
  }

  //! Functor used to access elements in the matrix
  //! \param i row index
  //! \param j column index
  //! \return element in position ij of the matrix
  T operator()(size_t i, size_t j) const {
    validateIndexes(i, j);
    return mData[i * mCols + j];
  }
  //endregion
  //endregion

  //! Returns a matrix filled with a single value
  //! \param rows number of rows in the matrix
  //! \param cols number of columns in the matrix
  //! \param value value to be used for initialization
  //! \return a matrix with all values set to <code>value</code>
  static Matrix fill(size_t rows, size_t cols, double value) {
    Matrix result(rows, cols, vector<T>(rows * cols, value));
    return result;
  }

  //! Creates a square matrix with a fixed value on the diagonal
  //! \param size dimensions of the square matrix
  //! \param value value to be used in the diagonal
  //! \return square matrix with a fixed value on the diagonal
  static Matrix diagonal(size_t size, double value) {
    Matrix result = zeros(size, size);
    for (size_t i = 0; i < size; i++)
      result(i, i) = value;

    return result;
  }

  bool isSquare() const {
    return mCols == mRows;
  }

  //! \return diagonal of the square matrix as a column vector
  Matrix diagonal() {
    if (!isSquare()) {
      throw runtime_error("Can't get the diagonal, not a square matrix");
    }

    Matrix result(mRows, 1);

    #pragma omp parallel
    for (size_t i = 0; i < mRows; i++)
      result(i, 0) = operator()(i, i);

    return result;
  }

  //! Returns the identity matrix
  //! \param size dimensions of the square matrix
  //! \return identity matrix
  static Matrix identity(size_t size) {
    return diagonal(size, 1);
  }

  //! Returns a matrix filled with ones
  //! \param rows number of rows in the matrix
  //! \param cols number of columns in the matrix
  //! \return matrix filled with ones
  static Matrix ones(size_t rows, size_t cols) {
    return fill(rows, cols, 1);
  }

  //! Returns a matrix filled with zeros
  //! \param rows number of rows in the matrix
  //! \param cols number of columns in the matrix
  //! \return matrix filled with zeros
  static Matrix zeros(size_t rows, size_t cols) {
    return fill(rows, cols, 0);
  }

  //! Executes the Hadamard, or entrywise multiplication between two matrices
  //! \param b The other matrix
  //! \return result of the Hadamard multiplication of the two matrices
  Matrix hadamard(const Matrix &b) {
    if (mCols != b.mCols || mRows != b.mRows)
      throw invalid_argument(
          "Cannot multiply these matrices element-wise: L = " + to_string(mRows) + "x" +
              to_string(mCols) + ", R = " + to_string(b.mRows) + "x" + to_string(b.mCols));

    Matrix result(mRows, mCols);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < mRows; i++) {
      for (size_t j = 0; j < mCols; j++) {
        result(i, j) = operator()(i, j) * b(i, j);
      }
    }

    return result;
  }

  //! Returns a submatrix of the current matrix, removing one row and column of the original matrix
  //! \param row index of the row to be removed
  //! \param column index of the column to be removed
  //! \return submatrix of the current matrix, with one less row and column
  Matrix submatrix(size_t row, size_t column) const {
    Matrix result(mRows - 1, mCols - 1);

    size_t subi = 0;

    #pragma omp parallel for
    for (size_t i = 0; i < mRows; i++) {
      size_t subj = 0;
      if (i == row) continue;
      for (size_t j = 0; j < mCols; j++) {
        if (j == column) continue;
        result(subi, subj) = operator()(i, j);
        subj++;
      }
      subi++;
    }

    return result;
  }

  //! Returns the minor of a matrix, which is the determinant of a submatrix
  //! where a single row and column are removed
  //! \param row index of the row to be removed
  //! \param column index of the column to be removed
  //! \return minor of the current matrix
  double getMinor(size_t row, size_t column) const {
//        the minor of a 2x2 a b is d c
//                           c d    b a
    if (mRows == 2 and mCols == 2) {
      Matrix result(2, 2);
      result(0, 0) = operator()(1, 1);
      result(0, 1) = operator()(1, 0);
      result(1, 0) = operator()(0, 1);
      result(1, 1) = operator()(0, 0);
      return result.determinant();
    }

    return submatrix(row, column).determinant();
  }

  //! Calculates the cofactor of a matrix at a given point
  //! \param row index of the row where the cofactor will be calculated
  //! \param column index of the column where the cofactor will be calculated
  //! \return cofactor of the matrix at the given position
  double cofactor(size_t row, size_t column) const {
    double minor;

    // special case for when our matrix is 2x2
    if (mRows == 2 and mCols == 2) {
      if (row == 0 and column == 0)
        minor = operator()(1, 1);
      else if (row == 1 and column == 1)
        minor = operator()(0, 0);
      else if (row == 0 and column == 1)
        minor = operator()(1, 0);
      else if (row == 1 and column == 0)
        minor = operator()(0, 1);
    } else
      minor = this->getMinor(row, column);
    return (row + column) % 2 == 0 ? minor : -minor;
  }

  //! Calculates the cofactor matrix
  //! \return Cofactor matrix of the current matrix
  Matrix cofactorMatrix() const {
    Matrix result(mRows, mCols);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < mRows; i++) {
      for (size_t j = 0; j < mCols; j++) {
        result(i, j) = cofactor(i, j);
      }
    }
    return result;
  }

  //! Returns the adjugate of the current matrix, which is the transpose of its cofactor matrix
  //! \return Adjugate of the current matrix
  Matrix adjugate() const {
    return cofactorMatrix().transpose();
  }

  //! Calculates the inverse of the current matrix. Raises an error if
  //! the matrix is singular, that is, its determinant is equal to 0
  //! \return inverse of the current matrix
  Matrix inverse() const {
    if (!isSquare())
      throw runtime_error("Cannot invert a non-square matrix");

    double det = determinant();

    if (det == 0)
      throw runtime_error("Matrix is singular");

    Matrix adj = adjugate();
    return adjugate() / det;
  };

  //! Calculates the determinant of the matrix
  //! \return determinant of the matrix
  double determinant() const {
    if (!isSquare()) {
      throw runtime_error("Cannot calculate the determinant of a non-square matrix");
    }

    size_t n = mRows;
    double d = 0;
    if (n == 2) {
      return ((operator()(0, 0) * operator()(1, 1)) -
          (operator()(1, 0) * operator()(0, 1)));
    } else {
      #pragma omp parallel for reduction (+:d)
      for (size_t c = 0; c < n; c++) {
        d += pow(-1, c) * operator()(0, c) * submatrix(0, c).determinant();
      }
      return d;
    }
  }

  //! Returns the transpose of a matrix
  //! \return transpose of the current matrix
  Matrix transpose() const {
    Matrix result(mCols, mRows);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < mRows; i++) {
      for (size_t j = 0; j < mCols; j++) {
        result(j, i) = operator()(i, j);
      }
    }

    return result;
  }

  //! Adds a column at the end of the matrix. Addition is done inplace.
  //! \param values a column vector containing the values to be added in the new column
  void addColumn(Matrix values) {
    addColumn(values, mCols);
  }

  //! Adds a row at the end of the matrix. Addition is done inplace.
  //! \param values a column vector containing the values to be added in the new row
  void addRow(Matrix values) {
    addRow(values, mRows);
  }

  //! Adds a column to the matrix at the given position. Addition is done inplace.
  //! \param values a column vector containing the values to be added in the new column
  //! \param position index of the new column. The column at the current
  //! position and all columns succeeding it are pushed forward.
  void addColumn(Matrix values, size_t position) {
    if (!isEmpty() and values.nRows() != mRows)
      throw invalid_argument("Wrong number of values passed for new column");
    if (values.nCols() != 1)
      throw invalid_argument("Can't add multiple columns at once");

    if (isEmpty()) {
      mRows = values.mRows;
      mCols = values.mCols;
      mData = values.mData;
      return;
    }

    vector<T> newData(mData.size() + values.mRows);

    size_t newData_mCols = mCols + 1;
    for (size_t i = 0; i < mRows; i++) {
      for (size_t j = 0; j < newData_mCols; j++) {
        if (j == position)
          newData[i * newData_mCols + j] = values(i, 0);
        else {
          int a = j > position;
          newData[i * newData_mCols + j] = operator()(i, j - (j > position));
        }
      }
    }
    mCols += 1;
    mData = newData;
  }

  //! Adds a row to the matrix at the given position. Addition is done inplace.
  //! \param values a column vector containing the values to be added in the new row
  //! \param position index of the new row. The row at the current
  //! position and all rows succeeding it are pushed forward.
  void addRow(Matrix values, size_t position) {
    if (!isEmpty() and values.mRows != mCols)
      throw invalid_argument("Wrong number of values passed for new row");
    if (values.mCols != 1)
      throw invalid_argument("Can't add multiple rows at once");

    if (isEmpty()) {
      mRows = values.mCols;
      mCols = values.mRows;
      mData = values.mData;
      return;
    }

    // TODO addColumn with same logic was wrong, must check this one

    vector<T> newData(mData.size() + values.mRows);

    mRows += 1;
    for (size_t i = 0; i < mRows; i++) {
      for (size_t j = 0; j < mCols; j++) {
        if (i == position)
          newData[i * mCols + j] = values(j, 0);
        else
          newData[i * mCols + j] = operator()(i - (i > position), j);
      }
    }

    mData = newData;
  }

  //! Removes a column from the matrix. Removal is done inplace.
  //! \param position index of the column to be removed.
  void removeColumn(int position) {
    // this is how you stop a reverse for loop with unsigned integers
    for (size_t i = mRows - 1; i != (size_t) -1; i--)
      mData.erase(mData.begin() + (i * mCols + position));

    mCols -= 1;
  }

  //! Returns only unique values from the matrix
  //! \return column vector containing the unique values from the matrix
  Matrix unique() const {
    // include all data from the inner vector in a set
    set<T> s;
    unsigned long size = mData.size();

    for (unsigned i = 0; i < size; ++i)
      s.insert(mData[i]);

    // include all the data from the set back into a vector
    vector<T> auxVec;
    auxVec.assign(s.begin(), s.end());

    // return a column matrix with the unique elements
    return Matrix(auxVec.size(), 1, auxVec);
  }

  //! Sorts elements of the matrix inplace
  void sort() {
    // just sort the inner vector
    std::sort(mData.begin(), mData.end());
  }

  //! Sorts the elements of a matrix
  //! \param m the matrix whose elements will be sorted
  //! \return a matrix with the same shape as <code>m</code>, with its elements sorted
  static Matrix sort(Matrix m) {
    // copy the inner vector of the matrix passed as argument
    // and return a new matrix with the sorted inner vector
    vector<T> data = m.mData;
    std::sort(data.begin(), data.end());
    return Matrix(m.mRows, m.mCols, data);
  }

  //! Counts occurrences of elements in a matrix.
  //! \return matrix with two columns. The first contains unique instances of the elements in the original matrix.
  //! The second column contains occurrences of the elements in the first column.
  Matrix count() {
    Matrix result = unique();
    result.sort();

    result.addColumn(zeros(result.mRows, 1), 1);

    for (size_t i = 0; i < mRows; i++)
      for (size_t j = 0; j < mCols; j++)
        for (size_t g = 0; g < result.mRows; g++)
          if (operator()(i, j) == result(g, 0)) {
            result(g, 1)++;
            break;
          }

    return result;
  }

  //! Calculates means of a matrix, grouped by classes
  //! \param groups a column vector containing group assignments
  //! \return a matrix containing as many columns as there are unique groups.
  //! Each column represents the mean of each group.
  //! Columns are sorted by the group numbers in ascending order.
  Matrix mean(Matrix groups) {
    if (mRows != groups.mRows)
      throw invalid_argument("Not enough groups for every element in the matrix");

    Matrix groupCount = groups.count();
    Matrix result = zeros(groupCount.mRows, mCols);

    for (size_t i = 0; i < mRows; i++) {
      for (size_t g = 0; g < groupCount.mRows; g++) {
        if (groups(i, 0) == groupCount(g, 0)) {
          for (size_t j = 0; j < mCols; j++) {
            result(g, j) += operator()(i, j);
          }
          break;
        }
      }
    }

    for (size_t i = 0; i < result.mRows; i++)
      for (size_t j = 0; j < result.mCols; j++)
        result(i, j) /= groupCount(i, 1);

    return result;
  }

  //! Calculates the mean of the columns of the matrix
  //! \return Column vector containing the means
  Matrix mean() {
    Matrix result = zeros(mCols, 1);

    for (size_t i = 0; i < mRows; i++) {
      for (size_t j = 0; j < mCols; j++) {
        result(j, 0) += operator()(i, j);
      }
    }

    result /= mRows;

    return result;
  }

  //! Calculates the scatter matrix. Columns are taken as features.
  //! \return scatter matrix
  Matrix scatter() {
    Matrix means = mean();
    Matrix result(mCols, mCols);

    for (size_t i = 0; i < mRows; i++) {
      Matrix rowDiff = getRow(i) - means;
      result += rowDiff * rowDiff.transpose();
    }

    return result;
  }

  //! Calculates the covariance matrix of the current matrix. Columns are taken as features.
  //! \return covariance matrix
  Matrix cov() {
    return scatter() / (mRows - 1);
  }

  //! Calculates the variance of the columns of the matrix
  //! \return Column vector containing the variances
  Matrix var() {
    Matrix means = mean();
    Matrix result = zeros(mCols, 1);

    for (size_t i = 0; i < mCols; i++) {
      for (size_t ii = 0; ii < mRows; ii++)
        result(i, 0) += pow((operator()(ii, i) - means(i, 0)), 2);

      result(i, 0) /= (mRows - 1);
    }

    return result;
  }

  //! Calculates the standard deviation of the columns of the matrix
  //! \return column vector containing standard deviations
  Matrix stdev() {
    Matrix result = var();

    #pragma omp parallel for
    for (size_t i = 0; i < mCols; i++)
      result(i, 0) = sqrt(result(i, 0));

    return result;
  }

  //! Reshapes the current matrix. The operation is done inplace.
  //! \param rows new number of rows
  //! \param cols new number of columns
  void reshape(size_t rows, size_t cols) {
    if (mData.size() != rows * cols)
      throw invalid_argument(
          "Invalid shape (" + to_string(rows) + "x" +
              to_string(cols) + " = " + to_string(rows * cols) +
              ") for a matrix with" + to_string(mData.size()) + " elements");

    mRows = rows;
    mCols = cols;
  }

  //! Gets a column from the matrix
  //! \param index index of the desired column
  //! \return column vector containing the values in the given column of the original matrix
  Matrix getColumn(size_t index) {
    if (index >= mCols)
      throw invalid_argument("Column index out of bounds");

    Matrix result(mRows, 1);
    #pragma omp parallel for
    for (size_t i = 0; i < mRows; i++)
      result(i, 0) = operator()(i, index);

    return result;
  }

  //! Gets a row from the matrix
  //! \param index index of the desired row
  //! \return column vector containing the values in the given row of the original matrix
  Matrix getRow(size_t index) {
    if (index >= mRows)
      throw invalid_argument("Row index out of bounds");

    Matrix result(mCols, 1);
    #pragma omp parallel for
    for (size_t i = 0; i < mCols; i++)
      result(i, 0) = operator()(index, i);

    return result;
  }

  //! Prints a matrix
  //! \param os output stream
  //! \param matrix the matrix to be printed
  //! \return output stream with the string representation of the matrix
  friend ostream &operator<<(ostream &os, const Matrix &matrix) {
    const int numWidth = 13;
    char fill = ' ';

    for (int i = 0; i < matrix.mRows; i++) {
      for (int j = 0; j < matrix.mCols; j++) {
        // the trick to print a table-like structure was stolen from here
        // https://stackoverflow.com/a/14796892
        os << left << setw(numWidth) << setfill(fill) << to_string(matrix(i, j));
      }
      os << endl;
    }

    return os;
  }

  //! Loads CSV data into a matrix
  //! \param path path of the CSV file
  //! \return a matrix constructed from the contents of the CSV file
  static Matrix fromCSV(const string &path) {
    vector<vector<double>> outer = CSVReader::csvToNumericVecVec(path, true);

    Matrix result(outer.size(), outer[0].size());

    for (size_t i = 0; i < result.mRows; i++)
      for (size_t j = 0; j < result.mCols; j++)
        result(i, j) = outer[i][j];

    return result;
  }

  //! Creates a diagonal matrix from a row or column vector
  //! \return diagonal matrix generated from the vector
  Matrix asDiagonal() {
    if (mRows != 1 and mCols != 1)
      throw runtime_error("Can't diagonalize, not a vector");

    size_t dimension = mCols > 1 ? mCols : mRows;

    Matrix result = zeros(dimension, dimension);

    #pragma omp parallel for
    for (size_t i = 0; i < dimension; i++) {
      result(i, i) = mCols > 1 ? operator()(0, i) : operator()(i, 0);
    }
    return result;
  }

  //! Returns a copy of the matrix
  //! \return copy of the current matrix
  Matrix copy() {
    Matrix result(mRows, mCols);
    result.mData = mData;
    return result;
  }

  //! Standardizes the columns of the matrix, subtracting each element of a column
  //! by the column mean and dividing it by the standard deviation of the column.
  //! \return a new matrix with the columns standardized as described
  Matrix standardize() {
    return standardize(mean(), stdev());
  }

  //! Standardizes the columns of the matrix, subtracting each element of a column
  //! by the <code>mean</code> argument and dividing it by the <code>stds</code> argument.
  //! \param means a column matrix containing the elements that will subtract each column of the original matrix
  //! \param stds a column matrix containing the elements that will divide each column of the original matrix
  //! \return a new matrix with the columns standardized as described
  Matrix standardize(Matrix means, Matrix stds) {
    if (!means.isColumn())
      throw invalid_argument("Argument \"mean\" must have exactly one column");
    if (!stds.isColumn())
      throw invalid_argument("Argument \"stds\" must have exactly one column");
    if (means.mRows != mCols)
      throw invalid_argument("Number of mean values is different than number of features");
    if (stds.mRows != mCols)
      throw invalid_argument("Number of std. dev. values is different than number of features");

    Matrix result(mRows, mCols);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < mRows; i++) {
      for (size_t j = 0; j < mCols; j++) {
        result(i, j) = (operator()(i, j) - means(j, 0)) / stds(j, 0);
      }
    }

    return result;
  }

  Matrix minusMean() {
    Matrix result(mRows, mCols), means = mean();

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < mRows; i++) {
      for (size_t j = 0; j < mCols; j++) {
        result(i, j) = operator()(i, j) - means(j, 0);
      }
    }

    return result;
  }

  //! Checks if the matrix contains a value
  //! \param value the vaue to look for
  //! \return true if the matrix contains the value, otherwise false
  bool contains(T value) {
    return std::find(mData.begin(), mData.end(), value) != mData.end();
  }

  //! Checks if the matrix is empty or uninitialized
  //! \return true if the matrix is empty or uninitialized, otherwise false
  bool isEmpty() {
    return mCols == 0 and mRows == 0;
  }

  //! Selects a subset of either columns or rows of the matrix
  //! \param bin a column vector containing only 0s and 1s, where indices with
  //! 1s indicate the indices of the columns/rows that will be returned by the method
  //! \param columns true if the filter will select the columns of the matrix, otherwise, rows will be selected
  //! \return
  Matrix filter(const Matrix<int> bin, bool columns = false) {
    size_t dimension = columns ? mCols : mRows;

    if (bin.nCols() != 1)
      throw invalid_argument("Binary filter must have only one column");
    if (bin.nRows() != dimension)
      throw invalid_argument("Binary filter has the wrong number of row entries");

    Matrix result;

    for (size_t i = 0; i < bin.nRows(); i++) {
      if (bin(i, 0)) {
        if (columns)
          result.addColumn(getColumn(i));
        else
          result.addRow(getRow(i));
      }
    }

    return result;
  }

  //! Selects a subset of rows of the matrix
  //! \param bin a column vector containing only 0s and 1s, where indices with
  //! 1s indicate the indices of the columns/rows that will be returned by the method
  Matrix getRows(const Matrix<int> bin) {
    return filter(bin);
  }

  //! Selects a subset of columns of the matrix
  //! \param bin a column vector containing only 0s and 1s, where indices with
  //! 1s indicate the indices of the columns/rows that will be returned by the method
  Matrix getColumns(const Matrix<int> bin) {
    return filter(bin, true);
  }

  //! Checks if the matrix is symmetric. A matrix is symmetric if it is equal to its transpose
  //! \return true if it is symmetric, otherwise false
  bool isSymmetric() {
    return *this == transpose();
  }

  //! Normalizes the column vectors of the matrix. Normalization is done by dividing
  //! each element of a vector by the length of the vector.
  //! \return Matrix with each column normalized by its length.
  Matrix normalize() {
    Matrix result(mRows, mCols, mData);

    // Calculate length of the column vector
    for (size_t j = 0; j < mCols; j++) {
      T length = 0;
      #pragma omp parallel for reduction(+:length)
      for (size_t i = 0; i < mRows; i++) {
        length += pow(result(i, j), 2);
      }
      length = sqrt(length);

      // divide each element of the column by its length
      for (size_t i = 0; i < mRows; i++) {
        result(i, j) /= length;
      }
    }

    return result;
  }

  //! Calculates the eigenvalues and eigenvectors of a matrix
  //! \param sort whether to sort the eigenvalues and eigenvectors
  //! \return a pair containing eigenvalues in a column vector in the first element of the pair
  //! and the correponding eigenvectors in the second element
  pair<Matrix, Matrix> eigen() {
    return isSymmetric() ? eigenSymmetric() : eigenNonSymmetric();
  };

  //! Calculates the eigenvalues and eigenvectors of a symmetric matrix using the Jacobi eigenvalue algorithm
  //! \return a pair containing eigenvalues in a column vector in the first element of the pair
  //! and the correponding eigenvectors in the second element
  pair<Matrix, Matrix> eigenSymmetric() {
    // Jacobi eigenvalue algorithm as explained
    // by profs Marina Andretta and Franklina Toledo

    // copy the current matrix
    Matrix A = copy();
    // initialize the eigenvector matrix
    Matrix V = identity(A.mCols);

    // get the tolerance
    double eps = numeric_limits<double>::epsilon();
    unsigned iterations = 0;

    // initiate the loop for numerical approximation of the eigenvalues
    while (true) {
      // find the element in the matrix with the largest modulo
      size_t p, q;
      T largest = 0;
      for (size_t i = 0; i < A.mRows; i++) {
        for (size_t j = 0; j < A.mCols; j++) {
          // it can't be in the diagonal
          if (i != j and abs(A(i, j)) > largest) {
            largest = abs(A(i, j));
            p = i;
            q = j;
          }
        }
      }

      // if the largest non-diagonal element of A is zero +/- eps,
      // it means A is almost diagonalized and the eigenvalues are
      // in the diagonal of A
      if (largest < 2 * eps or iterations >= 1000) {
        //eigenvalues are returned in a column matrix for convenience
        return eigsort(A.diagonal(), V);
      }

      iterations++;

      // else, perform a Jacobi rotation using this angle phi as reference
      double phi = (A(q, q) - A(p, p)) / (2 * A(p, q));
      double sign = (phi > 0) - (phi < 0);
      double t = phi == 0 ? 1 : 1 / (phi + sign * sqrt(pow(phi, 2) + 1));
      double cos = 1 / (sqrt(1 + pow(t, 2)));
      double sin = t / (sqrt(1 + pow(t, 2)));

      // the matrix that will apply the rotation is basically an identity matrix...
      Matrix U = identity(A.mRows);

      // ... with the exception of these values
      U(p, p) = U(q, q) = cos;
      U(p, q) = sin;
      U(q, p) = -sin;

      // apply the rotation
      A = U.transpose() * A * U;
      // update the corresponding eigenvectors
      V = V * U;
    }
  }

  //! Calculates the eigenvalues and eigenvectors of a on-symmetric matrix
  //! \param sort whether to sort the eigenvalues and eigenvectors
  //! \param calculate the eigenvalues and eigenvectors of a Hessenberg matrix with
  //! identical eigenvalues and eigenvectors instead
  //! \return a pair containing eigenvalues in a column vector in the first element of the pair
  //! and the correponding eigenvectors in the second element
  pair<Matrix, Matrix> eigenNonSymmetric(bool hes = true) {

    MatDoub unholyConvertion(static_cast<int>(mRows), static_cast<int>(mCols));

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < mRows; i++) {
      for (size_t j = 0; j < mCols; j++) {
        unholyConvertion[i][j] = operator()(i, j);
      }
    }

    Unsymmeig nr3Miracle(unholyConvertion, true, hes);

    Matrix eigenvalues(1, static_cast<size_t>(nr3Miracle.wri.size()));

    for (size_t i = 0; i < nr3Miracle.wri.size(); i++) {
      eigenvalues(0, i) = nr3Miracle.wri[i].real();
    }

    Matrix eigenvectors(static_cast<size_t>(nr3Miracle.zz.nrows()),
                        static_cast<size_t>(nr3Miracle.zz.ncols()));

    for (size_t i = 0; i < nr3Miracle.zz.nrows(); i++)
      for (size_t j = 0; j < nr3Miracle.zz.ncols(); j++)
        eigenvectors(i, j) = nr3Miracle.zz[i][j];

    return make_pair(eigenvalues, eigenvectors);
  }

  Matrix WithinClassScatter(Matrix y) {
    Matrix Sw = zeros(mCols, mCols);
    Matrix uniqueClasses = y.unique();

    for (size_t i = 0; i < uniqueClasses.nRows(); i++) {
      Matrix classElements = getRows(y == i); // get class elements

      Matrix scatterMatrix = classElements.scatter();
      Sw += scatterMatrix;
    }

    return Sw;
  }

  Matrix BetweenClassScatter(Matrix y) {
    Matrix innerMean = mean(y); // means for each class
    Matrix grandMean = mean(); // mean of the entire data set
    Matrix Sb = zeros(mCols, mCols);
    Matrix uniqueClasses = y.unique();

    for (size_t i = 0; i < uniqueClasses.nRows(); i++) {
      Matrix classElements = getRows(y == i); // get class elements
      Matrix meanDiff = innerMean.getRow(i) - grandMean;
      Sb += classElements.nRows() * meanDiff * meanDiff.transpose();
    }

    return Sb;
  }

  T sum() const {
    T sum_of_elems = 0;
    for (T n : mData)
      sum_of_elems += n;

    return sum_of_elems;
  }

  bool isColumn() const {
    return mCols == 1;
  }

  bool isRow() const {
    return mRows == 1;
  }

  T min() const {
    return *std::min_element(std::begin(mData), std::end(mData));
  }

  T max() const {
    return *std::max_element(std::begin(mData), std::end(mData));
  }

  Matrix<T> apply(function<T(T)> f) {
    Matrix<T> result(mRows, mCols, vector<T>(mRows * mCols, 0));
    std::transform(mData.begin(), mData.end(), result.mData.begin(), f);
    return result;
  }

  Matrix oneHot() {
    Matrix uniqueValues = unique();
    Matrix oneHotUnique = identity(uniqueValues.mRows);
    Matrix result(mRows, oneHotUnique.mCols);

    for (size_t i = 0; i < mRows; i++) {
      for (size_t ii = 0; ii < uniqueValues.mRows; ++ii) {
        if (getRow(i) == uniqueValues.getRow(ii)) {
          result.setRow(i, oneHotUnique.getRow(ii).transpose());
        }
      }
    }

    return result;
  }

  void setRow(size_t index, Matrix<T> row) {
    if (mRows < index)
      throw invalid_argument("Invalid row index, matrix is not that large");
    if (mCols != row.mCols)
      throw invalid_argument("Incompatible number of columns");
    if (row.mRows > 1)
      throw invalid_argument("Row matrix contains more than one row");

    for (size_t col = 0; col < mCols; col++)
      operator()(index, col) = row(0, col);
  }

  void setColumn(size_t index, Matrix<T> column) {
    if (mCols < index)
      throw invalid_argument("Invalid row column, matrix is not that large");
    if (mRows != column.mRows)
      throw invalid_argument("Incompatible number of rows");
    if (column.mCols > 1)
      throw invalid_argument("Column matrix contains more than one column");

    for (size_t row = 0; row < mCols; row++)
      operator()(row, index) = column(row, 0);
  }

  bool isBinary() const {
    Matrix<T> uniqueBin = unique();
    return uniqueBin.mRows <= 2 && uniqueBin.contains(1) or uniqueBin.contains(0);
  }
};

typedef Matrix<double> MatrixD;
typedef Matrix<int> MatrixI;

#endif //MACHINE_LEARNING_MATRIX_HPP
