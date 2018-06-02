#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <array>
#include <chrono>
#include "../Matrix.hpp"

using namespace std;
using myClock = chrono::high_resolution_clock;

void testMatrixFromCSV() {
  MatrixD m = MatrixD::fromCSV("../include/csv_reader/test/alpswater.csv");
  cout << m;
}

void testAddRowColumn() {
  const double arr[] = {1, 2,
                        3, 4,
                        5, 6};
  MatrixD m(3, 2, vector<double>(arr, arr + sizeof(arr) / sizeof(arr[0])));

  cout << m;
  const double zerosColumn[] = {0, 0, 0};
  MatrixD column(3, 1, vector<double>(zerosColumn, zerosColumn + sizeof(zerosColumn) / sizeof(zerosColumn[0])));
  m.addColumn(column, 1);
  cout << m;
//    m.addRow(0, vector<double>(zerosColumn, zerosColumn + sizeof(zerosColumn) / sizeof(zerosColumn[0])));
//    cout << m;
}

void testInverseDeterminant() {
  const double arr1[] = {3, 5, 2,
                         8, 4, 8,
                         2, 4, 7};
  MatrixD test_d1(3, 3, vector<double>(arr1, arr1 + sizeof(arr1) / sizeof(arr1[0])));
  const double arr2[] = {9, 5, 2, 5,
                         9, 5, 3, 7,
                         6, 5, 4, 8,
                         1, 5, 3, 7};
  MatrixD test_d2(4, 4, vector<double>(arr2, arr2 + sizeof(arr2) / sizeof(arr2[0])));
  const double arr3[] = {3, 6, 2,
                         8, 6, 5,
                         9, 1, 6};
  MatrixD test_d3(3, 3, vector<double>(arr3, arr3 + sizeof(arr3) / sizeof(arr3[0])));

  double d1 = test_d1.determinant(), d2 = test_d2.determinant(), d3 = test_d3.determinant();
  cout << d1 << endl;
  cout << d2 << endl;
  cout << d3 << endl;
  cout << test_d1.inverse();
  cout << test_d2.inverse();
  cout << test_d3.inverse();
}

void testOperations() {
  const double arr[] = {1, 2,
                        3, 4,
                        5, 6};
  MatrixD m(3, 2, vector<double>(arr, arr + sizeof(arr) / sizeof(arr[0]))),
      t = m.transpose(), mt = m * t, tm = t * m, mm = m + m, tt = t + t;
  cout << m;
  cout << t;
  cout << mt;
  cout << tm;
  cout << mm;
  cout << tt;
}

void testBigOperations() {
  // good for testing OpenMP implementation
  MersenneTwister twister;
  vector<size_t> sizes = {10, 50, 100, 250, 500, 1000};

  for (auto s: sizes) {
    MatrixD a(s, s, twister.vecFromNormal(s * s));
    MatrixD b(s, s, twister.vecFromNormal(s * s));

    chrono::time_point<chrono::system_clock> start = myClock::now();
    a * b;
    chrono::duration<float> execution_time = myClock::now() - start;
    cout << s << '\t' << execution_time.count() << endl;
  }
}

void testEigen() {
  const double arr[] = {4, 2, 0,
                        2, 5, 3,
                        0, 3, 6};
  MatrixD coitada(3, 3, vector<double>(arr, arr + sizeof(arr) / sizeof(arr[0])));

  pair<MatrixD, MatrixD> eigens = coitada.eigen();
  MatrixD val = eigens.first, vec = eigens.second;

  cout << val << endl << vec;
}

void sanityCheck() {
  const double arr[] = {9, 1, 1, 2,
                        9, 2, 3, 4,
                        9, 3, 5, 2,
                        9, 4, 7, 4};
  const double y[] = {0, 1, 0, 1};
  MatrixD m1(4, 4, vector<double>(arr, arr + sizeof(arr) / sizeof(arr[0])));
  MatrixD groups(4, 1, vector<double>(y, y + sizeof(y) / sizeof(y[0])));

  cout << m1.mean();
  cout << m1.mean(groups);
}

int main() {
  cout.precision(12);
  testAddRowColumn();
  testInverseDeterminant();
  testMatrices();
  testOperations();
  testBigOperations();
  testEigen();
  sanityCheck();
  return 0;
}
