#pragma once
#if defined(CUDA_ENABLE)
#define __CUDA_DEVICE__ __device__
#define __CUDA_HOST_DEVICE__ __host__ __device__
#define __CUDA_GLOBAL__ __global__

#else
#define __CUDA_DEVICE__ 
#define __CUDA_HOST_DEVICE__ 
#define __CUDA_GLOBAL__ 
#endif

#include <stdio.h>
#include <stdlib.h>

#if defined(CUDA_ENABLE)

#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#endif

int allocateNgbPreWVec(double*& ngb_pre_w_vec, int mask_side, int map_a, size_t total_size);
int allocateMaskWMat(double*& mask_w_mat, int mask_side, int map_a, size_t total_size);
int allocatePreW(double*& pre_w, int map_x, int map_y, int map_a);
void printMaskWMat(double* mask_w_mat, int mask_side, int map_a);
void printWG(double* pre_w, int map_x, int map_y, int map_a);
int cublasFunc(void);
