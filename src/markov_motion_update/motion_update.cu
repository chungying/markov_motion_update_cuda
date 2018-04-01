#include "markov_motion_update/motion_update.h"
#define IDX2M(x,y,d2) ((x)*(d2)+(y))
#define IDX2G(x,y,a,d2,d3) (((x)*(d2)*(d3))+(y)*(d3)+(a))
#define IDX2Q(opa,x,y,a,d2,d3,d4) (((opa)*(d2)*(d3)*(d4))+(x)*(d3)*(d4)+(y)*(d4)+(a))
int allocateNgbPreWVec(double* ngb_pre_w_vec, int map_x, int map_y, int map_a);
int allocateMaskWMat(double* mask_w_mat, int map_x, int map_y, int map_a);
int allocatePreW(double* pre_w, int map_x, int map_y, int map_a);

int main(void)
{
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  int map_x = 10;
  int map_y = 6;
  int map_a = 4;
  int mask_side = 3;
  int activ_no = 20;
  //prerequisite
  double* pre_w = 0;//previous weight vector, map_a*map_x*map_y
  if(allocatePreW(pre_w, map_x, map_y, map_a)!=0)
    return EXIT_FAILURE;
  double* cur_w = 0;//previous weight vector, map_a*map_x*map_y
  if(allocatePreW(cur_w, map_x, map_y, map_a)!=0)
    return EXIT_FAILURE;
  //host memory pointer
  double* mask_w_mat = 0;
  if(allocateMaskWMat(mask_w_mat, map_x, map_y, map_a)!=0)
    return EXIT_FAILURE;
  double* ngb_pre_w_vec = 0;
  if(allocateNgbPreWVec(ngb_pre_w_vec, map_x, map_y, map_a)!=0)
    return EXIT_FAILURE;
  //device memory pointer
  double* mask_w_mat_dev;//mask weight matrix, map_a by map_a*mask_side*mask_side
  double* ngb_pre_w_vec_dev;//neighbors' previous weight vector, map_a*mask_side*mask_side
  double ptk_w = 0;//k-th particle's weight at time t
  double* cur_w_dev = 0;
  cudaStat = cudaMalloc ((void**)&mask_w_mat_dev, map_a*map_a*mask_side*mask_side*sizeof(*mask_w_mat));
  if (cudaStat != cudaSuccess)
  {
      printf ("device memory allocation failed\n");
      cudaFree (mask_w_mat_dev);
      return EXIT_FAILURE;
  }
  cudaStat = cudaMalloc ((void**)&ngb_pre_w_vec_dev, map_a*mask_side*mask_side*sizeof(*ngb_pre_w_vec));
  if (cudaStat != cudaSuccess)
  {
      printf ("device memory allocation failed\n");
      cudaFree (ngb_pre_w_vec_dev);
      return EXIT_FAILURE;
  }
  cudaStat = cudaMalloc ((void**)&cur_w_dev, map_x*map_y*map_a*sizeof(*cur_w));
  if (cudaStat != cudaSuccess)
  {
      printf ("device memory allocation failed\n");
      cudaFree (cur_w_dev);
      return EXIT_FAILURE;
  }

  //cublas
  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS)
  {
      printf ("CUBLAS initialization failed\n");
      return EXIT_FAILURE;
  }
  stat = cublasSetVector (map_a*map_a*mask_side*mask_side, sizeof(*mask_w_mat), mask_w_mat, 1, mask_w_mat_dev, 1);
  if (stat != CUBLAS_STATUS_SUCCESS)
  {
    printf ("data download failed\n");
    cudaFree (mask_w_mat_dev);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  stat = cublasSetVector (map_a*mask_side*mask_side  , sizeof(*ngb_pre_w_vec), ngb_pre_w_vec, 1, ngb_pre_w_vec_dev, 1);
  if (stat != CUBLAS_STATUS_SUCCESS)
  {
    printf ("data download failed\n");
    cudaFree (mask_w_mat_dev);
    cudaFree (ngb_pre_w_vec_dev);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  //stat = cublasSetVector (map_x*map_y*map_a, sizeof(*cur_w), cur_w, 1, cur_w_dev, 1);

  for(int pidx = 0; pidx < map_x*map_y*map_a; pidx++)
  {

  }

  cudaFree (mask_w_mat_dev);
  cudaFree (ngb_pre_w_vec_dev);
  cudaFree (cur_w_dev);
  free(pre_w);
  free(cur_w);
  free(mask_w_mat);
  free(ngb_pre_w_vec);

}

int allocateNgbPreWVec(double* ngb_pre_w_vec, int map_x, int map_y, int map_a)
{
  ngb_pre_w_vec = (double *)malloc( map_a*mask_side*mask_side*sizeof(*ngb_pre_w_vec) );
  if (!ngb_pre_w_vec )
  {
    printf ("host memory allocation failed\n");
    return 1;
  }
  printf("ngb_pre_w_vec");
  for (i = 0; i < map_x; i++)
  {
    for (j = 0; j < map_y; j++)
    {
      for (k = 0; k < map_a; k++)
      {
        ngb_pre_w_vec[IDX2G(i,j,k,map_y,map_a)] = (double)(IDX2G(i,j,k,map_y,map_a));
        printf ("%7.0f", ngb_pre_w_vec[IDX2G(i,j,k,map_y,map_a)]);
      }
      printf("\n");
    }
    printf("\n");
  }
  return 0;
}

int allocateMaskWMat(double* mask_w_mat, int map_x, int map_y, int map_a)
{
  int h,i,j,k;
  mask_w_mat = (double *)malloc( map_a*map_x*map_y*map_a*sizeof(*mask_w_mat) );
  if (!mask_w_mat )
  {
    printf ("host memory allocation failed\n");
    return 1;
  }
  printf("mask_w_mat\n");
  for (h = 0; h < map_a; h++)
  {
    for (i = 0; i < map_x; i++)
    {
      for (j = 0; j < map_y; j++)
      {
        for (k = 0; k < map_a; k++)
        {
          mask_w_mat[IDX2Q(h,i,j,k,map_x,map_y,map_a)] = (double)(IDX2Q(h,i,j,k,map_x,map_y,map_a));
          printf ("%7.0f", mask_w_mat[IDX2Q(h,i,j,k,map_x,map_y,map_a)]);
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }
  return 0;
}

int allocatePreW(double* pre_w, int map_x, int map_y, int map_a)
{
  int i,j,k;
  pre_w = (double *)malloc( map_x*map_y*map_a*sizeof(*pre_w));
  if(!pre_w)
  {
    printf("host memory allocation failed\n");
    return 1;
  }
  printf("pre_w\n");
  for (i = 0; i < map_x; i++)
  {
    for (j = 0; j < map_y; j++)
    {
      for (k = 0; k < map_a; k++)
      {
        pre_w[IDX2G(i,j,k,map_y,map_a)] = (double)IDX2G(i,j,k,map_y,map_a);
        printf ("%7.0f", pre_w[IDX2G(i,j,k,map_y,map_a)]);
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("\n");

}
