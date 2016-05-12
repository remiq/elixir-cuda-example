/* Stand-alone CUDA code. No CNode here.
 * Example taken from http://developer.download.nvidia.com/books/cuda-by-example/cuda-by-example-sample.pdf
 */

#include <stdio.h>
#define N 10

__global__ void add( int *a, int *b, int *c ) {
  int tid = blockIdx.x;
  if (tid < N)
    c[tid] = a[tid] + b[tid];
}

int main(void) {
  int a[N], b[N], c[N];
  int *dev_a, *dev_b, *dev_c;

  cudaMalloc( (void**)&dev_a, N * sizeof(int) );
  cudaMalloc( (void**)&dev_b, N * sizeof(int) );
  cudaMalloc( (void**)&dev_c, N * sizeof(int) );

  for (int i=0; i<N; i++) {
    a[i] = -i;
    b[i] = i * i;
  }

  cudaMemcpy( dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice );
  cudaMemcpy( dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice );

  add<<<N,1>>>( dev_a, dev_b, dev_c );

  cudaMemcpy( c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost );

  for (int i=0; i<N; i++) {
    printf( "%d + %d = %d\n", a[i], b[i], c[i] );
  }

  cudaFree( dev_a );
  cudaFree( dev_b );
  cudaFree( dev_c );

  return 0;
}
