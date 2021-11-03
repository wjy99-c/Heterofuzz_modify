#include <stdio.h>
#include "kernel.h"

int kernel(char* argv){
    int a, b;
    char op;
    printf("%s\n",argv);
    FILE* f = fopen(argv, "r");
    fscanf(f, "%d\n%c\n%d\n", &a, &op, &b);
    printf("%d%c%d\n",a,op,b);
    

    data_t A[I][K];
    data_t B[K][J];
    data_t C[I][J];
    data_t C_dsa[I][J];

    for (int i = 0; i < I; i++){
      for (int k = 0; k < K; k++) {
        fscanf(f,"%f",&A[i][k]);
        printf("%f ",A[i][k]);
    }
    printf("\n");
    }
    for (int k = 0; k < K; k++){
      for (int j = 0; j < J; j++) {
        fscanf(f,"%f",&B[k][j]);
        printf("%f ",B[k][j]);
    }
    printf("\n");
    }
    fclose(f);
    for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++) {
      C[i][j] = 0;
      for (int k = 0; k < K; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }

    dsa_kernel(A, B, C_dsa);

  // comparison
  int err = 0;
  float thres = 0.001;
  for (int i = 0; i < I; i++) 
    for (int j = 0; j < J; j++) {
      if (fabs(C_dsa[i][j] - C[i][j]) > thres) {
        err++;
      }
    }

  if (err) {
    printf("Test failed with %d errors!\n", err);
    return -1;
  } else {
    printf("Test passed!\n");
    return 0;
  }
}

int main(int argc, char *argv[])
{
    int result;

    result = kernel(argv[1]);
    
    printf("%d\n", result);
    return 0;
}
