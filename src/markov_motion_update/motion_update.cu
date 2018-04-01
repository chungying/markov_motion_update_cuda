#include "markov_motion_update/motion_update.h"
#define IDX2M(x,y,d2) ((x)*(d2)+(y))
#define IDX2G(a,x,y,d2,d3) (((a)*(d2)*(d3))+(IDX2M(x,y,d3)))
#define IDX2Q(opa,a,x,y,d2,d3,d4) (((opa)*(d2)*(d3)*(d4))+(a)*(d3)*(d4)+(x)*(d4)+(y))
int allocateNgbPreWVec(double* ngb_pre_w_vec, int mask_side, int map_a);
int allocateMaskWMat(double* mask_w_mat, int mask_side, int map_a);
int allocatePreW(double* pre_w, int map_x, int map_y, int map_a);
void printWG(double* pre_w, int map_x, int map_y, int map_a);
void showMemInfo()
{
        // show memory usage of GPU

  size_t free_byte ;
  size_t total_byte ;
  
  cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
  
  if ( cudaSuccess != cuda_status )
  {
    printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
    exit(1);
  }
  double free_db = (double)free_byte ;
  double total_db = (double)total_byte ;
  double used_db = total_db - free_db ;
  printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n", used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}

int main(void)
{
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  int map_x = 4;
  int map_y = 4;
  int map_a = 4;
  int mask_side = 3;
  //prerequisite
  double* pre_w = 0;//previous weight vector, map_a*map_x*map_y
  if(allocatePreW(pre_w, map_x, map_y, map_a)!=0)
    return EXIT_FAILURE;
  double* cur_w = 0;//previous weight vector, map_a*map_x*map_y
  if(allocatePreW(cur_w, map_x, map_y, map_a)!=0)
    return EXIT_FAILURE;
  //host memory pointer
  double* mask_w_mat = 0;
  if(allocateMaskWMat(mask_w_mat, mask_side, map_a)!=0)
    return EXIT_FAILURE;
  double* ngb_pre_w_vec = 0;
  if(allocateNgbPreWVec(ngb_pre_w_vec, mask_side, map_a)!=0)
    return EXIT_FAILURE;
  //device memory pointer
  double* mask_w_mat_dev;//mask weight matrix, map_a by map_a*mask_side*mask_side
  double* ngb_pre_w_vec_dev;//neighbors' previous weight vector, map_a*mask_side*mask_side
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
  printf ("CUBLAS initialization succeed\n");
  stat = cublasSetVector (map_a*mask_side*mask_side*map_a, sizeof(*mask_w_mat), mask_w_mat, 1, mask_w_mat_dev, 1);
  if (stat != CUBLAS_STATUS_SUCCESS)
  {
    printf ("mask_w_mat_dev download failed, status: %d, element size: %d, total mem: %d B, size:%d\n", stat,sizeof(*mask_w_mat), map_a*map_a*mask_side*mask_side*sizeof(*mask_w_mat),map_a*mask_side);
    showMemInfo();
    cudaFree (mask_w_mat_dev);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  printf ("mask_w_mat_dev download succeed\n");
  stat = cublasSetVector (map_a*mask_side*mask_side  , sizeof(*ngb_pre_w_vec), ngb_pre_w_vec, 1, ngb_pre_w_vec_dev, 1);
  if (stat != CUBLAS_STATUS_SUCCESS)
  {
    printf ("ngb_pre_w_vec_dev download failed, status: %d\n", stat);
    cudaFree (mask_w_mat_dev);
    cudaFree (ngb_pre_w_vec_dev);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  printf ("ngb_pre_w_vec_dev download succeed\n");
  //stat = cublasSetVector (map_x*map_y*map_a, sizeof(*cur_w), cur_w, 1, cur_w_dev, 1);

  for(int pidx_x = 0; pidx_x < map_x; pidx_x++)
  for(int pidx_y = 0; pidx_y < map_y; pidx_y++)
  for(int pidx_a = 0; pidx_a < map_a; pidx_a++)
  {
    //For each particle
    int pidx = IDX2G(pidx_a, pidx_x, pidx_y, map_x, map_y);
    //TODO get ngb_pre_w_vec with active position
    //TODO cp ngb_pre_w_vec to device
    for(int mask_x = 0 ; mask_x < mask_side ; mask_x++)
    for(int mask_y = 0; mask_y < mask_side ; mask_y++)
    for(int mask_a = 0; mask_a < map_a ; mask_a++)
    {
      int ngb_pidx_x = pidx_x - (mask_side-1)/2 + mask_x;
      int ngb_pidx_y = pidx_y - (mask_side-1)/2 + mask_y;
      int ngb_pidx_a = mask_a;
      int ngb_pidx = IDX2G(ngb_pidx_a, ngb_pidx_x, ngb_pidx_y, map_x, map_y);
      ngb_pre_w_vec[IDX2G(mask_a,mask_x,mask_y,mask_side,mask_side)] = pre_w[ngb_pidx];
    }
    stat = cublasSetVector (map_a*mask_side*mask_side  , sizeof(*ngb_pre_w_vec), ngb_pre_w_vec, 1, ngb_pre_w_vec_dev, 1);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
      printf ("data download failed\n");
      cudaFree (ngb_pre_w_vec_dev);
      cudaFree (mask_w_mat_dev);
      cudaFree (cur_w_dev);
      cublasDestroy(handle);
      return EXIT_FAILURE;
    }
    //TODO get origin_particle_angle_idx
    int origin_particle_angle_idx = (pidx+1)%map_a;//assume this value instead of ANG2IDX(sample[pidx].v[2])
    //TODO cur_w_dev[pidx] = cublasDDot(mask_w_mat_dev[origin_particle_angle_idx], ngb_pre_w_vec)
    cublasDdot(handle, map_x*map_y*map_a, ngb_pre_w_vec_dev, 1, &mask_w_mat_dev[origin_particle_angle_idx], 1, &cur_w_dev[pidx]);
  }
  cublasGetVector(map_x*map_y*map_a, sizeof(*cur_w_dev), cur_w_dev, 1, cur_w, 1);
  printf("cur_w\n");
  printWG(cur_w, map_x, map_y, map_a);

  cudaFree (mask_w_mat_dev);
  cudaFree (ngb_pre_w_vec_dev);
  cudaFree (cur_w_dev);
  free(pre_w);
  free(cur_w);
  free(mask_w_mat);
  free(ngb_pre_w_vec);

}

int allocateNgbPreWVec(double* ngb_pre_w_vec, int mask_side, int map_a)
{
  ngb_pre_w_vec = (double *)malloc( mask_side*mask_side*map_a*sizeof(*ngb_pre_w_vec) );
  if (!ngb_pre_w_vec )
  {
    printf ("host memory allocation failed\n");
    return 1;
  }
  return 0;
}

int allocateMaskWMat(double* mask_w_mat, int mask_side, int map_a)
{
  int h,i,j,k;
  mask_w_mat = (double *)malloc( map_a*map_a*mask_side*mask_side*sizeof(*mask_w_mat) );
  if (!mask_w_mat )
  {
    printf ("host memory allocation failed\n");
    return 1;
  }
  printf("mask_w_mat\n");
  for (h = 0; h < map_a; h++)
  {
    for (i = 0; i < mask_side; i++)
    {
      for (j = 0; j < mask_side; j++)
      {
        for (k = 0; k < map_a; k++)
        {
          mask_w_mat[IDX2Q(h,k,i,j,map_a,mask_side,mask_side)] = (double)(IDX2Q(h,k,i,j,map_a,mask_side,mask_side));
          printf ("%7.0f", mask_w_mat[IDX2Q(h,k,i,j,map_a,mask_side,mask_side)]);
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }
  return 0;
}

void printWG(double* pre_w, int map_x, int map_y, int map_a)
{
  int i,j,k;
  for (k = 0; k < map_a; k++)
  {
    for (j = 0; j < map_y; j++)
    {
      for (i = 0; i < map_x; i++)
      {
        pre_w[IDX2G(k,i,j,map_x,map_y)] = (double)IDX2G(k,i,j,map_x,map_y);
        printf ("%7.0f", pre_w[IDX2G(k,i,j,map_x,map_y)]);
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("\n");
  ;
}

int allocatePreW(double* pre_w, int map_x, int map_y, int map_a)
{
  pre_w = (double *)malloc( map_x*map_y*map_a*sizeof(*pre_w));
  if(!pre_w)
  {
    printf("host memory allocation failed\n");
    return 1;
  }
  printf("pre_w\n");
  printWG(pre_w, map_x, map_y, map_a);
  return 0;
}
