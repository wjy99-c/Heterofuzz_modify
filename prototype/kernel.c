#include "kernel.h"

void dsa_kernel(data_t A[I][K], data_t B[K][J], data_t C[I][J]) {
  data_t A_ext[I][J][K];
  data_t B_ext[I][J][K];
  data_t C_ext[I][J][K];

  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++) {
      for (int k = 0; k < K; k++) {
        if (j == 0) {
          A_ext[i][j][k] = A[i][k];
        } else {
          A_ext[i][j][k] = A_ext[i][j - 1][k];
        }

        if (i == 0) {
          B_ext[i][j][k] = B[k][j];
        } else {
          B_ext[i][j][k] = B_ext[i - 1][j][k];
        }

        if (k == 0) {
          C_ext[i][j][k] = A_ext[i][j][k] * B_ext[i][j][k];
        } else {
          C_ext[i][j][k] = C_ext[i][j][k - 1]  + A_ext[i][j][k] * B_ext[i][j][k];
        }
      }
      C[i][j] = C_ext[i][j][K - 1];
    }
}
