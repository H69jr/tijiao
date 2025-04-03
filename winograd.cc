#include <omp.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include "utils.h"
#include <stdint.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include "winograd.h"

#define WINOGRAD_ALPHA 6
#define WINOGRAD_TILE 4

// 在文件头部添加CUDA内核函数声明
__global__ void batch_image_transform_kernel(
    float* packed_images, float* V, 
    int batch_size, tiling_info_t ti, 
    V_shape_t vs, int collapsed_dim_size);

__global__ void batch_filter_transform_kernel(
    float* packed_filters, float* U,
    int num_filters, filter_shape_t fs,
    U_shape_t us, int collapsed_dim_size);

__global__ void batch_output_transform_kernel(
    float* M, float* Y,
    int batch_size, tiling_info_t ti,
    out_shape_t os, int collapsed_dim_size);

__constant__ float Bt[WINOGRAD_ALPHA][WINOGRAD_ALPHA] = {
    {4.0f,  0.0f, -5.0f,  0.0f, 1.0f, 0.0f},
    {0.0f, -4.0f, -4.0f,  1.0f, 1.0f, 0.0f},
    {0.0f,  4.0f, -1.0f, -1.0f, 1.0f, 0.0f},
    {0.0f, -2.0f, -1.0f,  2.0f, 1.0f, 0.0f},
    {0.0f,  2.0f, -1.0f, -2.0f, 1.0f, 0.0f},
    {0.0f,  4.0f,  0.0f, -5.0f, 0.0f, 1.0f}
};

//分块信息计算函数
tiling_info_t get_tiling_info(const image_shape_t is, const filter_shape_t fs) {
    tiling_info_t ti;
    
    // Winograd F(6x6,3x3)参数
    const int alpha = 6;
    const int tile_size = 4;  // 有效输出块大小
    
    // 输入分块
    ti.tile_w = (is.w + tile_size - 1) / tile_size;
    ti.tile_h = (is.h + tile_size - 1) / tile_size;
    ti.tile_in_w = alpha;
    ti.tile_in_h = alpha;
    
    // 输出分块
    ti.tile_out_w = tile_size;
    ti.tile_out_h = tile_size;
    
    // 其他计算
    ti.num_tiles = ti.tile_w * ti.tile_h;
    ti.tiles_per_row = ti.tile_w;
    
    // 内存尺寸计算
    ti.input_w = is.w;
    ti.input_h = is.h;
    ti.input_c = is.ic;
    
    return ti;
}

// 全局cuBLAS句柄
cublasHandle_t cublas_handle;

// 初始化函数 (程序开始时调用一次)
void init_cublas() {
    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);
}

// 清理函数 (程序结束时调用)
void cleanup_cublas() {
    cublasDestroy(cublas_handle);
}


// 添加image_packing需要的结构体
typedef struct {
    float *data;
    int64_t stride[4];
} Tensor4D;

static Tensor4D create_tensor(int64_t h, int64_t w, int64_t tile, int64_t ic) {
    Tensor4D tensor;
    tensor.stride[0] = w * tile * ic;
    tensor.stride[1] = tile * ic;
    tensor.stride[2] = ic;
    tensor.stride[3] = 1;
    size_t size = h * w * tile * ic * sizeof(float);
    if (posix_memalign((void **)&tensor.data, 64, size) != 0) {
        perror("posix_memalign failed");
        exit(EXIT_FAILURE);
    }
    return tensor;
}

static inline float *get_element(Tensor4D *tensor, int64_t h, int64_t w, int64_t tile, int64_t ic) {
    return &tensor->data[h * tensor->stride[0] + w * tensor->stride[1] + tile * tensor->stride[2] + ic * tensor->stride[3]];
}

// 添加filter_packing需要的结构体
typedef struct {
    float *data;
    int64_t stride[4]; // [oc][ic][h][w]
} FilterTensor;



static FilterTensor create_filter_tensor(int64_t oc, int64_t ic, int64_t h, int64_t w) {
    FilterTensor tensor;
    tensor.stride[0] = ic * h * w; // oc步长
    tensor.stride[1] = h * w;      // ic步长
    tensor.stride[2] = w;          // h步长
    tensor.stride[3] = 1;          // w步长
    size_t size = oc * ic * h * w * sizeof(float);
    if (posix_memalign((void **)&tensor.data, 64, size) != 0) {
        perror("posix_memalign failed");
        exit(EXIT_FAILURE);
    }
    return tensor;
}

static inline float *get_filter_element(FilterTensor *tensor, int64_t oc, int64_t ic, int64_t h, int64_t w) {
    return &tensor->data[oc * tensor->stride[0] + ic * tensor->stride[1] + h * tensor->stride[2] + w * tensor->stride[3]];
}

// 分配GPU内存-GPU内存管理函数
float* gpu_alloc(size_t size) {
    float *d_ptr = NULL;
    cudaError_t err = cudaMalloc(&d_ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return d_ptr;
}

// 释放GPU内存
void gpu_free(float *d_ptr) {
    cudaFree(d_ptr);
}

// 主机到设备拷贝
void host_to_device(float *h_src, float *d_dst, size_t size) {
    cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice);
}

// 设备到主机拷贝
void device_to_host(float *d_src, float *h_dst, size_t size) {
    cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost);
}

void image_transform(float *__restrict__ packed_image,
                    float *__restrict__ V,
                    const V_shape_t vs,
                    const tiling_info_t ti,
                    const int64_t collapsed_dim_size)
{
    // 定义张量视图（兼容原始指针和多维数组访问）
    typedef float(*packed_image_tensor_t)[ti.tile_in_w][collapsed_dim_size];
    typedef float(*V_tensor_t)[ti.tile_in_w][collapsed_dim_size];
    packed_image_tensor_t packed_image_tensor = (packed_image_tensor_t)packed_image;
    V_tensor_t V_tensor = (V_tensor_t)V;

    // Winograd变换的临时变量
    float z0, z1, z2, z3, z4, z5, z6;

    // 外层循环：按高度分块 (h)
    for (int64_t h = 0; h < ti.tile_in_h; ++h) {
        // OpenMP并行化：每个线程处理不同的idx和w组合
        #pragma omp parallel for private(z0, z1, z2, z3, z4, z5, z6)
        for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
            for (int64_t w = 0; w < ti.tile_in_w; ++w) {
                // 预取下一位置数据（优化缓存）
                __builtin_prefetch(&packed_image_tensor[h][w+1][idx], 0, 3);

                // --- Winograd F(6x6, 3x3) 输入变换 ---
                // 阶段1：读取并计算前两行
                z6 = packed_image_tensor[h][w][idx];
                z0 = 4.0f * z6;

                z6 = packed_image_tensor[h+1][w][idx];
                z1 = -4.0f * z6;
                z2 = 4.0f * z6;
                z3 = -2.0f * z6;
                z4 = 2.0f * z6;
                z5 = 4.0f * z6;


                // 阶段2：处理第三行
                z6 = packed_image_tensor[h+2][w][idx];
                z0 += -5.0f * z6;
                z1 += -4.0f * z6;
                z2 += -1.0f * z6;
                z3 += -z6;
                z4 += -z6;

                // 阶段3：处理第四行
                z6 = packed_image_tensor[h+3][w][idx];
                z1 += z6;
                z2 += -z6;
                z3 += 2.0f * z6;
                z4 += -2.0f * z6;
                z5 += -5.0f * z6;

                // 阶段4：处理第五行
                z6 = packed_image_tensor[h+4][w][idx];
                z0 += z6;
                z1 += z6;
                z2 += z6;
                z3 += z6;
                z4 += z6;

                // 阶段5：处理第六行
                z6 = packed_image_tensor[h+5][w][idx];
                z5 += z6;

                // 写入变换结果到V张量
                V_tensor[h][w][idx] = z0;
                V_tensor[h+1][w][idx] = z1;
                V_tensor[h+2][w][idx] = z2;
                V_tensor[h+3][w][idx] = z3;
                V_tensor[h+4][w][idx] = z4;
                V_tensor[h+5][w][idx] = z5;
            }
        }
    }
}




void filter_transform(float *__restrict__ packed_filter,
                      float *__restrict__ U,
                      const filter_shape_t fs,
                      const U_shape_t us,
                      const int64_t collapsed_dim_size) {
  typedef float(*packed_filter_tensor_t)[fs.w][collapsed_dim_size];
  typedef float(*U_tensor_t)[us.w][collapsed_dim_size];
  packed_filter_tensor_t packed_filter_tensor = (packed_filter_tensor_t)packed_filter;
  U_tensor_t U_tensor = (U_tensor_t)U;

  float z0, z1, z2, z3, z4, z5, z6;

  #pragma omp parallel for 
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
    for (int64_t w = 0; w < fs.w; ++w) {
       __builtin_prefetch( &packed_filter_tensor[0][w+1][idx],0,3);
      z6 = packed_filter_tensor[0][w][idx];

      z0 = (1.0f / 4.0f) * z6;
      z1 = (-1.0f / 6.0f) * z6;
      z2 = (-1.0f / 6.0f) * z6;
      z3 = (1.0f / 24.0f) * z6;
      z4 = (1.0f / 24.0f) * z6;

      z6 = packed_filter_tensor[1][w][idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (1.0f / 6.0f) * z6;
      z3 += (1.0f / 12.0f) * z6;
      z4 += (-1.0f / 12.0f) * z6;

      z6 = packed_filter_tensor[2][w][idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (-1.0f / 6.0f) * z6;
      z3 += (1.0f / 6.0f) * z6;
      z4 += (1.0f / 6.0f) * z6;
      z5 = z6;

      U_tensor[0][w][idx] = z0;
      U_tensor[1][w][idx] = z1;
      U_tensor[2][w][idx] = z2;
      U_tensor[3][w][idx] = z3;
      U_tensor[4][w][idx] = z4;
      U_tensor[5][w][idx] = z5;
    }

    for (int64_t h = 0; h < us.h; ++h) {
      z6 = U_tensor[h][0][idx];

      z0 = (1.0f / 4.0f) * z6;
      z1 = (-1.0f / 6.0f) * z6;
      z2 = (-1.0f / 6.0f) * z6;
      z3 = (1.0f / 24.0f) * z6;
      z4 = (1.0f / 24.0f) * z6;

      z6 = U_tensor[h][1][idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (1.0f / 6.0f) * z6;
      z3 += (1.0f / 12.0f) * z6;
      z4 += (-1.0f / 12.0f) * z6;

      z6 = U_tensor[h][2][idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (-1.0f / 6.0f) * z6;
      z3 += (1.0f / 6.0f) * z6;
      z4 += (1.0f / 6.0f) * z6;
      z5 = z6;

      U_tensor[h][0][idx] = z0;
      U_tensor[h][1][idx] = z1;
      U_tensor[h][2][idx] = z2;
      U_tensor[h][3][idx] = z3;
      U_tensor[h][4][idx] = z4;
      U_tensor[h][5][idx] = z5;
    }
  }
}


void output_transform(float *__restrict__ M,
                      float *__restrict__ Y,
                      const tiling_info_t ti,
                      const int64_t collapsed_dim_size) {
  typedef float(*M_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  typedef float(*Y_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  M_tensor_t M_tensor = (M_tensor_t)M;
  Y_tensor_t Y_tensor = (Y_tensor_t)Y;
  float z0, z1, z2, z3, z4;

  #pragma omp parallel for 
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
    //#pragma omp simd
    for (int64_t w = 0; w < ti.tile_in_w; ++w) {
      z4 = M_tensor[0][w][idx];
      z0 = z4;

      z4 = M_tensor[1][w][idx];
      z0 = z0 + z4;
      z1 = z4;
      z2 = z4;
      z3 = z4;

      z4 = M_tensor[2][w][idx];
      z0 += z4;
      z1 += -z4;
      z2 += z4;
      z3 += -z4;

      z4 = M_tensor[3][w][idx];
      z0 += z4;
      z1 += 2.0f * z4;
      z2 += 4.0f * z4;
      z3 += 8.0f * z4;

      z4 = M_tensor[4][w][idx];
      z0 += z4;
      z1 += -2.0f * z4;
      z2 += 4.0f * z4;
      z3 += -8.0f * z4;

      z4 = M_tensor[5][w][idx];
      z3 += z4;

      Y_tensor[0][w][idx] = z0;
      Y_tensor[1][w][idx] = z1;
      Y_tensor[2][w][idx] = z2;
      Y_tensor[3][w][idx] = z3;
    }

    for (int64_t h = 0; h < ti.tile_out_h; ++h) {
      z4 = Y_tensor[h][0][idx];

      z0 = z4;

      z4 = Y_tensor[h][1][idx];
      z0 += z4;
      z1 = z4;
      z2 = z4;
      z3 = z4;

      z4 = Y_tensor[h][2][idx];
      z0 += z4;
      z1 += -z4;
      z2 += z4;
      z3 += -z4;

      z4 = Y_tensor[h][3][idx];
      z0 += z4;
      z1 += 2.0f * z4;
      z2 += 4.0f * z4;
      z3 += 8.0f * z4;

      z4 = Y_tensor[h][4][idx];
      z0 += z4;
      z1 += -2.0f * z4;
      z2 += 4.0f * z4;
      z3 += -8.0f * z4;

      z4 = Y_tensor[h][5][idx];

      z3 += z4;

      Y_tensor[h][0][idx] = z0;
      Y_tensor[h][1][idx] = z1;
      Y_tensor[h][2][idx] = z2;
      Y_tensor[h][3][idx] = z3;
    }
  }
}


void filter_packing(float *__restrict__ filter, 
                    float *__restrict__ packed_filter, 
                    const filter_shape_t fs) {
    // 1. 创建临时FilterTensor（SoA布局）
    FilterTensor tmp_tensor = create_filter_tensor(fs.oc, fs.ic, fs.h, fs.w);

    // 2. 按优化顺序填充数据（oc→ic→h→w）
    #pragma omp parallel for collapse(3)
    for (int64_t oc = 0; oc < fs.oc; oc++) {
        for (int64_t ic = 0; ic < fs.ic; ic++) {
            for (int64_t h = 0; h < fs.h; h++) {
                for (int64_t w = 0; w < fs.w; w++) {
                    float *elem = get_filter_element(&tmp_tensor, oc, ic, h, w);
                    *elem = filter[oc * fs.ic * fs.h * fs.w + ic * fs.h * fs.w + h * fs.w + w];
                }
            }
        }
    }

    // 3. 拷贝回原始packed_filter（如需兼容原始布局）
    #pragma omp parallel for
    for (int64_t i = 0; i < fs.oc * fs.ic * fs.h * fs.w; i++) {
        packed_filter[i] = tmp_tensor.data[i];
    }

    // 4. 释放临时内存
    free(tmp_tensor.data);
}

void image_packing(float *__restrict__ image,
                   float *__restrict__ packed_image,
                   const image_shape_t is,
                   const tiling_info_t ti) {
    // 检查字段名是否匹配（根据报错提示修正）
    int64_t tile_in_h = ti.tile_in_h;  // 确保字段名正确
    int64_t tile_in_w = ti.tile_in_w;

    // 创建临时Tensor
    Tensor4D tmp_tensor = create_tensor(tile_in_h, tile_in_w, ti.num_tiles, is.ic);

    // 填充数据
    #pragma omp parallel for collapse(2)
    for (int64_t ic = 0; ic < is.ic; ic++) {
        for (int64_t tile = 0; tile < ti.num_tiles; tile++) {
            tile_index_t tidx = get_tile_index(tile, ti);
            for (int64_t h = 0; h < tile_in_h; h++) {
                for (int64_t w = 0; w < tile_in_w; w++) {
                    float *elem = get_element(&tmp_tensor, h, w, tile, ic);
                    int64_t src_h = tidx.th * 4 + h;
                    int64_t src_w = tidx.tw * 4 + w;
                    *elem = (src_h < is.h && src_w < is.w) ? 
                           image[tidx.b * is.ic * is.h * is.w + ic * is.h * is.w + src_h * is.w + src_w] : 0.0f;
                }
            }
        }
    }

    // 拷贝回原始数组
    #pragma omp parallel for
    for (int64_t i = 0; i < tile_in_h * tile_in_w * ti.num_tiles * is.ic; i++) {
        packed_image[i] = tmp_tensor.data[i];
    }

    // 释放内存
    free(tmp_tensor.data);
}

void output_unpacking_store(float *__restrict__ Y,
                            float *__restrict__ out,
                            const out_shape_t os,
                            const tiling_info_t ti) 
{
  typedef float(*Y_tensor_t)[ti.tile_in_w][os.oc][ti.num_tiles];
  typedef float(*out_tensor_t)[os.oc][os.h][os.w];
  Y_tensor_t Y_tensor = (Y_tensor_t)Y;
  out_tensor_t out_tensor = (out_tensor_t)out;

  for (int64_t oc = 0; oc < os.oc; oc++) {
    for (int64_t tile = 0; tile < ti.num_tiles; tile++) {
        tile_index_t tidx = get_tile_index(tile, ti);
        int64_t batch = tidx.b;
        for (int64_t h = 0; h < ti.tile_out_h; h++) {
            for (int64_t w = 0; w < ti.tile_out_w; w++) {
                int64_t dst_h = tidx.th * 4 + h;
                int64_t dst_w = tidx.tw * 4 + w;
                if (dst_h < os.h && dst_w < os.w) {
                    out_tensor[batch][oc][dst_h][dst_w] = Y_tensor[h][w][oc][tile];
                }
            }
        }
    }
  }
}

void sgemm_gpu(const int64_t M, const int64_t N, const int64_t K,
               float *d_A, float *d_B, float *d_C) {
    // 常量定义
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // 1. 分配半精度内存
    __half *d_A_fp16 = NULL, *d_B_fp16 = NULL;
    cudaError_t err;
    
    err = cudaMalloc((void**)&d_A_fp16, M * K * sizeof(__half));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate d_A_fp16: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMalloc((void**)&d_B_fp16, K * N * sizeof(__half));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate d_B_fp16: %s\n", cudaGetErrorString(err));
        cudaFree(d_A_fp16);
        exit(EXIT_FAILURE);
    }

    // 2. 创建转换流（与主计算流并行）
    cudaStream_t convertStream;
    cudaStreamCreate(&convertStream);
    
    // 3. 异步转换为fp16（使用独立流）
    cudaMemcpyAsync(d_A_fp16, d_A, M*K*sizeof(float), 
                   cudaMemcpyDeviceToDevice, convertStream);
    cudaMemcpyAsync(d_B_fp16, d_B, K*N*sizeof(float),
                   cudaMemcpyDeviceToDevice, convertStream);
    
    // 4. 等待转换完成
    cudaStreamSynchronize(convertStream);
    cudaStreamDestroy(convertStream);

    // 5. 执行混合精度GEMM
    cublasStatus_t status = cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_N, 
        CUBLAS_OP_N,
        N,    // 注意：cuBLAS使用列主序，N和M交换
        M,    // 以兼容原始行主序逻辑
        K,
        &alpha,
        d_B_fp16, CUDA_R_16F, N,
        d_A_fp16, CUDA_R_16F, K,
        &beta,
        d_C, CUDA_R_32F, N,
        CUDA_R_32F,            // 计算类型
        CUBLAS_GEMM_DEFAULT_TENSOR_OP  // 启用Tensor Core
    );

    // 6. 错误检查
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error: %d\n", status);
        cudaFree(d_A_fp16);
        cudaFree(d_B_fp16);
        exit(EXIT_FAILURE);
    }

    // 7. 释放临时内存
    cudaFree(d_A_fp16);
    cudaFree(d_B_fp16);

    // 8. 同步设备（可选，用于精确计时）
    cudaDeviceSynchronize();
}

// 实现矩阵乘法函数
void winograd_batched_sgemm(
    int64_t batch_count, int64_t M, int64_t N, int64_t K,
    float* d_V, float* d_U, float* d_M) 
{
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemmStridedBatched(
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,  // 注意行列主序转换
        &alpha,
        d_U, N, N*K,
        d_V, K, M*K,
        &beta,
        d_M, N, M*N,
        batch_count);
}

void winograd_convolution_gpu(
    float *d_image, 
    float *d_filter, 
    float *d_output,
    const image_shape_t is, 
    const filter_shape_t fs,
    const tiling_info_t ti) 
{
    
    // 1. 计算输出形状和变换矩阵形状
    out_shape_t os = get_output_shape(is, fs);
    U_shape_t us = get_U_shape(fs, ti);
    V_shape_t vs = get_V_shape(is, ti);

    // 2. 定义中间维度（根据Winograd变换规则）
    // 对于F(6x6,3x3)变换，中间矩阵维度为6x6
    const int64_t winograd_alpha = 6;  // Winograd变换矩阵大小
    const int64_t collapsed_dim_size = is.ic * fs.oc;  // 输入通道×输出通道

    // 3. 分配GPU内存
    size_t image_size = ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic * sizeof(float);
    size_t filter_size = fs.oc * fs.ic * fs.h * fs.w * sizeof(float);
    size_t output_size = os.oc * os.h * os.w * sizeof(float);

    float *d_packed_image = gpu_alloc(image_size);
    float *d_packed_filter = gpu_alloc(filter_size);
    float *d_V = gpu_alloc(ti.tile_in_h * ti.tile_in_w * collapsed_dim_size * sizeof(float));
    float *d_U = gpu_alloc(us.h * us.w * collapsed_dim_size * sizeof(float));
    float *d_M = gpu_alloc(ti.tile_out_h * ti.tile_out_w * collapsed_dim_size * sizeof(float));

// ---  计时开始（覆盖核心计算步骤）---
    auto start = std::chrono::high_resolution_clock::now();

    // 4. 执行变换和矩阵乘法
    // 输入变换（批处理）
    batch_image_transform(
        d_packed_image,
        d_V,
        is.bs,  // batch_size
        ti,
        vs,
        collapsed_dim_size);

    // 滤波器变换（批处理）
    batch_filter_transform(
        d_packed_filter,
        d_U,
        fs.oc,  // num_filters
        fs,
        us,
        collapsed_dim_size);

    // 矩阵乘法（使用cuBLAS批处理GEMM）
    const int64_t M = ti.tile_out_h * ti.tile_out_w;  // 输出tile大小
    const int64_t N = fs.oc;                         // 输出通道数
    const int64_t K = is.ic;                         // 输入通道数
    const int64_t batch_count = ti.num_tiles;        // tile数量

    winograd_batched_sgemm(
        batch_count,
        M, N, K,
        d_V, d_U, d_M);

    // 输出变换（批处理）
    batch_output_transform(
        d_M,
        d_output,
        is.bs,  // batch_size
        ti,
        os,
        collapsed_dim_size);

    // ---  计时结束并计算性能 ---
    auto end = std::chrono::high_resolution_clock::now();
    float elapsed_time = std::chrono::duration<float>(end - start).count() * 1000;  // 毫秒

    // 计算 GFlops（Winograd F(6x6,3x3) 的 flops 公式）
    float flops = 2.0f * fs.oc * is.ic * os.h * os.w * is.bs * (6 * 6 / (4 * 4));  // 根据实际算法调整
    float gflops = (flops / (elapsed_time * 1e6));  // 转换为 GFlops

    // 打印结果
    printf("Winograd Convolution: Elapse time %.6f ms. (%.2f GFlops)\n", elapsed_time, gflops);

    // 5. 释放GPU内存
    gpu_free(d_packed_image);
    gpu_free(d_packed_filter);
    gpu_free(d_V);
    gpu_free(d_U);
    gpu_free(d_M);
}

