module util;

import std.math;
import std.algorithm;
import std.random;


double sigmoid(double x) {
  return 1.0 / (1.0 + pow(E, -x));
}


double dsigmoid(double y) {
  return y * (1.0 - y);
}


double[][] makeMatrix(int rows, int colums, double value) {
  double[][] mat;
  mat.length = rows;
  for (int i; i < rows; ++i) {
    for(int j; j < colums; ++j) {
      mat[i] ~= value;
    }
  }
  return mat;
}


double[][] randomMatrix(int rows, int colums, double lower, double upper) {
  double[][] mat;
  mat.length = rows;
  for (int i; i < rows; ++i) {
    for(int j; j < colums; ++j) {
      mat[i] ~= uniform(0.0, cast(double) (upper - lower)) + lower;
    }
  }
  return mat;
}


int[] genRandomIdx(int n) {
  int[] arr;
  for (int i; i < n; ++i) {
    arr ~= i;
  }

  for (int i; i < n; ++i) {
    auto j = i + uniform(0, n-i);
    swap(arr[i], arr[j]);
  }
  return arr;
}

