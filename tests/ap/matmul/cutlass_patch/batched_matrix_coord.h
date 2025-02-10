#pragma once

#include "cutlass/cutlass.h"

namespace cutlass {

struct BatchedMatrixCoord {
  int batch;
  int row;
  int column;

  CUTLASS_HOST_DEVICE
  BatchedMatrixCoord() : batch(0), row(0), column(0) {}

  CUTLASS_HOST_DEVICE
  BatchedMatrixCoord(int b, int r, int c) : batch(b), row(r), column(c) {}
};

};
