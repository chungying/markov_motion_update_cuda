#include "markov_motion_update/motion_update.h"
#define IDX2M(x,y,d2) ((x)*(d2)+(y))
#define IDX2G(a,x,y,d2,d3) (((a)*(d2)*(d3))+(x)*(d3)+(y))

int main(void)
{
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  int map_a = 3;
  int map_x = 10;
  int map_y = 6;
  int mask_side = 3;
  int activ_no = 20;
  //device memory pointer
  double* dA;//previous weight grid, map_a by map_x by map_y
  double* dB;//current weight grid, map_a by map_x by map_y
  double* dC;//mask weight grid, map_a by mask_side by mask_side
  double* dD;//particle-weight vector, active_no 
  double* dE;//particle-neighbor-active matrix, active_no by (mask_side*mask_side)
  double* dF;//neighbor-weight matrix, active_no by (mask_side*mask_side)
  double* dG;//particle-weight vector, active_no
  //host memory pointer
  double* a = 0;
  double* b = 0;
  double* c = 0;
  a = (double *)malloc( map_x*map_y*map_a*sizeof(*a) );
  b = (double *)malloc( map_x*map_y*map_a*sizeof(*b) );
  c = (double *)malloc( mask_side*mask_side*map_a*sizeof(*c) );
  if (!a || !b || !c)
  {
      printf ("host memory allocation failed\n");
      return EXIT_FAILURE;
  }
  printf("original a\n");
  for (i = 0; i < map_x; i++)
  {
    for (j = 0; j < map_y; j++)
    {
      for (k = 0; k < map_a; k++)
      {
        a[IDX2C(i,j,k,map_x,map_y)] = (double)(IDX2C(i,j,k,map_x,map_y));
        printf ("%7.0f", a[IDX2C(i,j,k,map_x,map_y)]);
      }
      printf ("\n");
    }
  }
  printf("original b\n");
  for (i = 0; i < map_x; i++)
  {
    for (j = 0; j < map_y; j++)
    {
      for (k = 0; k < map_a; k++)
      {
        b[IDX2C(i,j,k,map_x,map_y)] = (double)(0);
        printf ("%7.0f", b[IDX2C(i,j,k,map_x,map_y)]);
      }
      printf ("\n");
    }
  }

  cudaStat = cudaMalloc ((void**)&dA, map_x*map_y*map_a*sizeof(*a));
  if (cudaStat != cudaSuccess)
  {
      printf ("device memory allocation failed\n");
      return EXIT_FAILURE;
  }
  cudaStat = cudaMalloc ((void**)&dB, map_x*map_y*map_a*sizeof(*b));
  if (cudaStat != cudaSuccess)
  {
      printf ("device memory allocation failed\n");
      return EXIT_FAILURE;
  }
  cudaStat = cudaMalloc ((void**)&dC, mask_side*mask_side*map_a*sizeof(*c));
  if (cudaStat != cudaSuccess)
  {
      printf ("device memory allocation failed\n");
      return EXIT_FAILURE;
  }

  //cublas
  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS)
  {
      printf ("CUBLAS initialization failed\n");
      return EXIT_FAILURE;
  }



}
