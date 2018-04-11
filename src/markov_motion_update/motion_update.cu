#include "markov_motion_update/motion_update.h"
#include "ros/ros.h"
#include <vector>
#include <numeric>
#define IDX2M(x,y,d2) ((x)*(d2)+(y))
#define IDX2G(a,x,y,d2,d3) (((a)*(d2)*(d3))+(IDX2M(x,y,d3)))
#define IDX2Q(opa,a,x,y,d2,d3,d4) (((opa)*(d2)*(d3)*(d4))+(a)*(d3)*(d4)+(x)*(d4)+(y))

double CPUdot(double*& A, double*& B, int size)
{
  double product = 0.0;
  for(int i = 0 ; i < size ; ++i)
    product += A[i] * B[i];
  return product;
}

int check(double*& vec, double*& bak, int size)
{
  for(int i = 0 ; i < size ; ++i)
  {
    if(vec[i] != bak[i])
      return 1;
  }
  return 0;
}

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

int cublasFunc(void)
{
  ros::Time::init();
  std::vector<int> vec_cuda1;
  std::vector<int> vec_cuda2;
  std::vector<int> vec_cuda3;
  std::vector<int> vec_cudaall;
  std::vector<int> vec_cpu;
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  int map_x = 700;
  int map_y = 500;
  int map_a = 72;
  int mask_side = 91;//note that this has to be an odd positive number
  int total_mem_allocated = 0;//GPU memory size allocated by this program in Byte
  //prerequisite
  double* pre_w = 0;//previous weight vector, map_a*map_x*map_y
  double* cur_w = 0;//previous weight vector, map_a*map_x*map_y
  double* single_w = 0;
  pre_w = (double *)malloc( map_x*map_y*map_a*sizeof(*pre_w));
  cur_w = (double *)malloc( map_x*map_y*map_a*sizeof(*cur_w));
  single_w = (double *)malloc( sizeof(*single_w));
  *single_w = 0;
  if(!pre_w || !cur_w)
  {
    printf("host memory allocation failed\n");
    return 1;
  }
  int i,j,k;
  for (k = 0; k < map_a; k++)
    for (j = 0; j < map_y; j++)
      for (i = 0; i < map_x; i++)
      {
        pre_w[IDX2G(k,i,j,map_x,map_y)] = (double)IDX2G(k,i,j,map_x,map_y);
        cur_w[IDX2G(k,i,j,map_x,map_y)] = 0;
      }
  //printf("pre_w\n");
  //printWG(pre_w, map_x, map_y, map_a);

  //host memory pointer
  double* mask_w_mat = 0;
  size_t ngb_pre_w_vec_size = map_a*mask_side*mask_side;
  size_t mask_w_mat_size = map_a*ngb_pre_w_vec_size;
  if(allocateMaskWMat(mask_w_mat, mask_side, map_a, mask_w_mat_size)!=0)
    return EXIT_FAILURE;
  double* ngb_pre_w_vec = 0;
  double* ngb_pre_w_vec_bak = 0;
  double* mask_w_vec_bak = 0;
  if(allocateNgbPreWVec(ngb_pre_w_vec, mask_side, map_a, ngb_pre_w_vec_size)!=0)
    return EXIT_FAILURE;
  if(allocateNgbPreWVec(ngb_pre_w_vec_bak, mask_side, map_a, ngb_pre_w_vec_size)!=0)
    return EXIT_FAILURE;
  if(allocateNgbPreWVec(mask_w_vec_bak, mask_side, map_a, ngb_pre_w_vec_size)!=0)
    return EXIT_FAILURE;
  //device memory pointer
  double* mask_w_mat_dev;//mask weight matrix, map_a by map_a*mask_side*mask_side
  double* ngb_pre_w_vec_dev;//neighbors' previous weight vector, map_a*mask_side*mask_side
  double* cur_w_dev = 0;
  showMemInfo();
  cudaStat = cudaMalloc ((void **)(&mask_w_mat_dev), mask_w_mat_size*sizeof(*mask_w_mat));
  total_mem_allocated += mask_w_mat_size*sizeof(*mask_w_mat);
  if (cudaStat != cudaSuccess)
  {
      printf ("device memory allocation failed\n");
      cudaFree (mask_w_mat_dev);
      return EXIT_FAILURE;
  }
  cudaStat = cudaMalloc ((void **)(&ngb_pre_w_vec_dev), ngb_pre_w_vec_size*sizeof(*ngb_pre_w_vec));
  total_mem_allocated += ngb_pre_w_vec_size*sizeof(*ngb_pre_w_vec);
  if (cudaStat != cudaSuccess)
  {
      printf ("device memory allocation failed\n");
      cudaFree (ngb_pre_w_vec_dev);
      return EXIT_FAILURE;
  }
  cudaStat = cudaMalloc ((void **)(&cur_w_dev), map_x*map_y*map_a*sizeof(*cur_w));
  total_mem_allocated += map_x*map_y*map_a*sizeof(*cur_w);
  if (cudaStat != cudaSuccess)
  {
      printf ("device memory allocation failed\n");
      cudaFree (cur_w_dev);
      return EXIT_FAILURE;
  }
  printf("allocated %d Byte of GPU memory\n",total_mem_allocated);
  //cublas
  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS)
  {
      printf ("CUBLAS initialization failed\n");
      return EXIT_FAILURE;
  }
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  printf ("CUBLAS initialization succeed\n");
  //printf("original cur_w\n");
  //printWG(cur_w, map_x, map_y, map_a);
  stat = cublasSetVector (map_x*map_y*map_a , sizeof(*cur_w_dev), cur_w, 1, cur_w_dev, 1);
  if (stat != CUBLAS_STATUS_SUCCESS)
  {
      printf ("cublasSetVector failed, %d\n",stat);
      return EXIT_FAILURE;
  }
  cublasGetVector(map_x*map_y*map_a, sizeof(*cur_w_dev), cur_w_dev, 1, cur_w, 1);
  //printf("cur_w_dev\n");
  //printWG(cur_w, map_x, map_y, map_a);
  stat = cublasSetVector (mask_w_mat_size, sizeof(*mask_w_mat), mask_w_mat, 1, mask_w_mat_dev, 1);
  if (stat != CUBLAS_STATUS_SUCCESS)
  {
    printf ("mask_w_mat_dev download failed, status: %d, element size: %ld, total mem: %ld B, size:%d\n", stat,sizeof(*mask_w_mat), mask_w_mat_size*sizeof(*mask_w_mat),map_a*mask_side);
    cudaFree (mask_w_mat_dev);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  //printf ("mask_w_mat_dev download succeed\n");
  //printMaskWMat(mask_w_mat, mask_side, map_a);
  stat = cublasGetVector (mask_w_mat_size, sizeof(*mask_w_mat), mask_w_mat_dev, 1, mask_w_mat, 1);
  if (stat != CUBLAS_STATUS_SUCCESS)
  {
    printf ("mask_w_mat upload failed, status: %d, element size: %ld, total mem: %ld B, size:%d\n", stat,sizeof(*mask_w_mat), mask_w_mat_size*sizeof(*mask_w_mat),map_a*mask_side);
    showMemInfo();
    cudaFree (mask_w_mat_dev);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  //printf ("mask_w_mat_dev upload succeed\nprint mask_w_mat again\n");
  //printMaskWMat(mask_w_mat, mask_side, map_a);
  int particle_count = 0;
  int runs = 100;
  for(int pidx_x = 0; pidx_x < map_x; pidx_x++)
  for(int pidx_y = 0; pidx_y < map_y; pidx_y++)
  for(int pidx_a = 0; pidx_a < map_a; pidx_a++)
  {
    if(particle_count >= runs)
    {
      pidx_x = map_x;//break outter loop
      pidx_y = map_y;//break outter loop
      break;
    }
    particle_count++;
    //For each particle
    int pidx = IDX2G(pidx_a, pidx_x, pidx_y, map_x, map_y);
    //TODO get ngb_pre_w_vec with active position
    //TODO cp ngb_pre_w_vec to device
    for(int mask_a = 0; mask_a < map_a ; mask_a++)
    {
      for(int mask_x = 0 ; mask_x < mask_side ; mask_x++)
      {
        for(int mask_y = 0; mask_y < mask_side ; mask_y++)
        {
          int ngb_pidx_x = pidx_x - (mask_side-1)/2 + mask_x;
          int ngb_pidx_y = pidx_y - (mask_side-1)/2 + mask_y;
          int ngb_pidx_a = mask_a;
          int ngb_pidx = IDX2G(ngb_pidx_a, ngb_pidx_x, ngb_pidx_y, map_x, map_y);
          int ngb_pre_w_vec_idx = IDX2G(mask_a,mask_x,mask_y,mask_side,mask_side);
          if(ngb_pre_w_vec_idx < 0 || ngb_pre_w_vec_idx > mask_side*mask_side*map_a)
            printf("wrong ngb_pre_w_vec_idx %d from %d %d %d\n",ngb_pre_w_vec_idx, mask_a, mask_x, mask_y);
          //printf("(%d, %d, %d)=%d->%d",mask_x,mask_y,mask_a,ngb_pre_w_vec_idx,ngb_pidx);
          if(ngb_pidx_x < 0 || ngb_pidx_x > map_x || 
             ngb_pidx_y < 0 || ngb_pidx_y > map_y || 
             ngb_pidx_a < 0 || ngb_pidx_a > map_a)
          {
            //printf(" skipped \n");
            ngb_pre_w_vec[ngb_pre_w_vec_idx] = 0.0;
          }
          else
          {
            //printf("\n");
            ngb_pre_w_vec[ngb_pre_w_vec_idx] = pre_w[ngb_pidx];
            //printf(" = %f, %f\n",ngb_pre_w_vec[ngb_pre_w_vec_idx], pre_w[ngb_pidx]);
          }
        }
      }
    }
    //printf("setting ngb_pre_w_vec\n");
    ros::Time cuda11 = ros::Time::now();
    stat = cublasSetVector(ngb_pre_w_vec_size , sizeof(*ngb_pre_w_vec), ngb_pre_w_vec, 1, ngb_pre_w_vec_dev, 1);
    ros::Time cuda12 = ros::Time::now();
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
      printf ("ngb_pre_w_vec data download failed\n");
      cudaFree (ngb_pre_w_vec_dev);
      cudaFree (mask_w_mat_dev);
      cudaFree (cur_w_dev);
      cublasDestroy(handle);
      return EXIT_FAILURE;
    }
    stat = cublasGetVector(ngb_pre_w_vec_size , sizeof(*ngb_pre_w_vec_bak), ngb_pre_w_vec_dev, 1, ngb_pre_w_vec_bak, 1);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
      printf ("ngb_pre_w_vec_bak data upload failed\n");
      return EXIT_FAILURE;
    }
    if(check(ngb_pre_w_vec, ngb_pre_w_vec_bak, ngb_pre_w_vec_size) != 0)
    {
      printNgbPreWVec(ngb_pre_w_vec, mask_side, map_a);
      printNgbPreWVec(ngb_pre_w_vec_bak, mask_side, map_a);
      printf("ngb_pre_w_vec and ngb_pre_w_vec_bak are not the same");
      return EXIT_FAILURE;
    }
    //get origin_particle_angle_idx
    int origin_particle_angle_idx = ((pidx+1)%map_a)*ngb_pre_w_vec_size;//assume this value instead of ANG2IDX(sample[pidx].v[2])
    //printf("executing cublasDdot\n");
    /* A*B=W
    ngb_pre_w_vec_dev        : A vector 
    mask_w_mat_dev           : a matrix consisting of all B vectors, where the number of B vectors is 
    mask_w_mat_size          : the number of all elements in the matrix
    origin_particle_angle_idx: the index of each B vector in the matrix
    cur_w_dev                : an array for storing a sequence of W
    pidx                     : the index of cur_w_dev indicating which position should be used for storing a result W
    */
    ros::Time cuda21 = ros::Time::now();
    cublasDdot(handle, ngb_pre_w_vec_size, ngb_pre_w_vec_dev, 1, (mask_w_mat_dev+origin_particle_angle_idx), 1, (cur_w_dev+pidx));
    ros::Time cuda22 = ros::Time::now();

    //vector B
    stat = cublasGetVector(ngb_pre_w_vec_size , sizeof(*mask_w_vec_bak), (mask_w_mat_dev+origin_particle_angle_idx), 1, mask_w_vec_bak, 1);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
      printf("upload mask_w_mat_dev+origin_particle_angle_idx failed\n");
      return EXIT_FAILURE;
    }
    ros::Time cuda31 = ros::Time::now();
    stat = cublasGetVector(1 , sizeof(*single_w), (cur_w_dev+pidx), 1, single_w, 1);
    ros::Time cuda32 = ros::Time::now();
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
      printf("upload  failed\n");
      return EXIT_FAILURE;
    }
    ros::Time cpu11 = ros::Time::now();
    double cpu_w = CPUdot( ngb_pre_w_vec_bak, mask_w_vec_bak, ngb_pre_w_vec_size);
    ros::Time cpu12 = ros::Time::now();
    if(cpu_w != *single_w)
    {
      printf("print vector A\n");
      printNgbPreWVec(ngb_pre_w_vec_bak, mask_side, map_a);
      printf("print vector B\n");
      printNgbPreWVec(mask_w_vec_bak, mask_side, map_a);
      printf("result W from GPU is %f\n",*single_w);
      printf("result W from CPU is %f\n", cpu_w);
      printf("results from CPU and GPU are not the same\n");
      return EXIT_FAILURE;
    }
    //else
    //  printf("results from CPU and GPU are the same\n");
    int cuda1 = (cuda12 - cuda11).toNSec();
    int cuda2 = (cuda22 - cuda21).toNSec();
    int cuda3 = (cuda32 - cuda31).toNSec();
    int cudaall = (cuda1 + cuda2 + cuda3);
    int cpu = (cpu12 - cpu11).toNSec();
    vec_cuda1.push_back(cuda1);
    vec_cuda2.push_back(cuda2);
    vec_cuda3.push_back(cuda3);
    vec_cudaall.push_back(cudaall);
    vec_cpu.push_back(cpu);
    //printf("GPU times: %ld + %ld + %ld = %ld\n", cuda1, cuda2, cuda3, cudaall);
    //printf("CPU dot time: %ld\n", cpu);
  }
  //printf("original cur_w\n");
  //printWG(cur_w, map_x, map_y, map_a);
  cublasGetVector(map_x*map_y*map_a, sizeof(*cur_w_dev), cur_w_dev, 1, cur_w, 1);
  //printf("cur_w\n");
  //printWG(cur_w, map_x, map_y, map_a);

  printf("size of computation: %ld\n", vec_cpu.size());
  printf("printing GPU mem info\n");
  showMemInfo();
  cudaFree (mask_w_mat_dev);
  cudaFree (ngb_pre_w_vec_dev);
  cudaFree (cur_w_dev);
  free(pre_w);
  free(cur_w);
  free(mask_w_mat);
  free(ngb_pre_w_vec);
  printf("printing GPU mem info after releasing\n");
  showMemInfo();
  printf("cuda 1 avg time  : %f ns\n", 1.0*std::accumulate(vec_cuda1.begin(), vec_cuda1.end(), 0)/vec_cuda1.size());
  printf("cuda 2 avg time  : %f ns\n", 1.0*std::accumulate(vec_cuda2.begin(), vec_cuda2.end(), 0)/vec_cuda2.size());
  printf("cuda 3 avg time  : %f ns\n", 1.0*std::accumulate(vec_cuda3.begin(), vec_cuda3.end(), 0)/vec_cuda3.size());
  printf("cuda all avg time: %f ns\n", 1.0*std::accumulate(vec_cudaall.begin(), vec_cudaall.end(), 0)/vec_cudaall.size());
  printf("cpu avg time     : %f ns\n", 1.0*std::accumulate(vec_cpu.begin(), vec_cpu.end(), 0)/vec_cpu.size());
  return EXIT_SUCCESS;
}

void printNgbPreWVec(double*& ngb_pre_w_vec, int mask_side, int map_a)
{
  for(int mask_a = 0 ; mask_a < map_a;mask_a++)
  {
  for(int mask_x = 0 ; mask_x < mask_side;mask_x++)
    {
  for(int mask_y = 0 ; mask_y < mask_side;mask_y++)
      {
        int ngb_pre_w_vec_idx = IDX2G(mask_a,mask_x,mask_y,mask_side,mask_side);
        printf ("%f ", ngb_pre_w_vec[ngb_pre_w_vec_idx]);
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("\n");
}

int allocateNgbPreWVec(double*& ngb_pre_w_vec, int mask_side, int map_a,size_t total_size)
{
  ngb_pre_w_vec = (double *)malloc( total_size*sizeof(*ngb_pre_w_vec) );
  if (!ngb_pre_w_vec )
  {
    printf ("host memory allocation failed\n");
    return 1;
  }
  return 0;
}

int allocateMaskWMat(double*& mask_w_mat, int mask_side, int map_a,size_t total_size)
{
  mask_w_mat = (double*)malloc( total_size*sizeof(*mask_w_mat) );
  if (!mask_w_mat )
  {
    printf ("host memory allocation failed\n");
    return 1;
  }
  int h,i,j,k;
  for (h = 0; h < map_a; h++)
    for (i = 0; i < mask_side; i++)
      for (j = 0; j < mask_side; j++)
        for (k = 0; k < map_a; k++)
          //mask_w_mat[IDX2Q(h,k,i,j,map_a,mask_side,mask_side)] = (double)(IDX2Q(h,k,i,j,map_a,mask_side,mask_side));
          mask_w_mat[IDX2Q(h,k,i,j,map_a,mask_side,mask_side)] = 2;
  return 0;
}

void printMaskWMat(double* mask_w_mat, int mask_side, int map_a)
{
  int h,i,j,k;
  printf("mask_w_mat\n");
  for (h = 0; h < map_a; h++)
  {
    for (i = 0; i < mask_side; i++)
    {
      for (j = 0; j < mask_side; j++)
      {
        for (k = 0; k < map_a; k++)
        {
          printf ("%7.0f", mask_w_mat[IDX2Q(h,k,i,j,map_a,mask_side,mask_side)]);
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }
}

void printWG(double* pre_w, int map_x, int map_y, int map_a)
{
  int i,j,k;
  for (k = 0; k < map_a; k++)
  {
    for (i = 0; i < map_x; i++)
    {
      for (j = 0; j < map_y; j++)
      {
        printf ("%f ", pre_w[IDX2G(k,i,j,map_x,map_y)]);
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("\n");
  ;
}

int allocatePreW(double*& pre_w, int map_x, int map_y, int map_a)
{
  pre_w = (double *)malloc( map_x*map_y*map_a*sizeof(*pre_w));
  if(!pre_w)
  {
    printf("host memory allocation failed\n");
    return 1;
  }
  int i,j,k;
  for (k = 0; k < map_a; k++)
    for (j = 0; j < map_y; j++)
      for (i = 0; i < map_x; i++)
        //pre_w[IDX2G(k,i,j,map_x,map_y)] = (double)IDX2G(k,i,j,map_x,map_y);
        pre_w[IDX2G(k,i,j,map_x,map_y)] = 1;
  //printf("pre_w\n");
  //printWG(pre_w, map_x, map_y, map_a);
  return 0;
}
