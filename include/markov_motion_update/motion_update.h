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

