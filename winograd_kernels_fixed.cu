#include <cuda_runtime.h>
#include "winograd.h"
#include "utils.h"

// 错误检查宏
#define CHECK_CUDA(call) \
do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

 // 确保包含核函数声明和必要的结构体定义（如 filter_shape_t, U_shape_t 等）

//-----------------------------------------------------------------------------
// CUDA 核函数实现
//-----------------------------------------------------------------------------

// 输入变换核函数
__global__ void batch_image_transform_kernel(
    float* packed_images,
    float* V,
    int batch_size,
    tiling_info_t ti,
    V_shape_t vs,
    int collapsed_dim_size)
{
    // 实现输入变换逻辑
    // 计算线程全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= collapsed_dim_size) return;
    V[idx] = packed_images[idx] * 2.0f;
    // 实际计算逻辑
}

// 滤波器变换核函数
__global__ void batch_filter_transform_kernel(
    float* packed_filters,
    float* U,
    int num_filters,
    filter_shape_t fs,
    U_shape_t us,
    int collapsed_dim_size)
{
    // 实现你的滤波器变换逻辑
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= collapsed_dim_size) return;

    // TODO: 填充实际计算逻辑
}

// 输出变换核函数
__global__ void batch_output_transform_kernel(
    float* M,
    float* Y,
    int batch_size,
    tiling_info_t ti,
    out_shape_t os,
    int collapsed_dim_size)
{
    // 实现你的输出变换逻辑
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= collapsed_dim_size) return;

    // TODO: 填充实际计算逻辑
}

//将输入数据批量传输到GPU，并调用批处理核函数
void batch_image_transform(
    float *d_packed_images, // GPU上的输入数据 (NHWC布局)
    float *d_V,            // GPU输出V矩阵
    int batch_size,        // 批处理大小
    const tiling_info_t ti,
    const V_shape_t vs,
    const int64_t  collapsed_dim_size)
{

    dim3 block(16, 16); // 每个线程块处理16x16的w和idx组合
    dim3 grid(
        (collapsed_dim_size + block.x - 1) / block.x,
        (ti.tile_in_w + block.y - 1) / block.y,
        batch_size * ti.tile_in_h // 批处理维度
    );

    batch_image_transform_kernel<<<grid, block>>>(
        d_packed_images, d_V, batch_size, ti, vs,collapsed_dim_size);
    cudaDeviceSynchronize();
}

// 批量处理滤波器
void batch_filter_transform(
    float *d_packed_filters, // GPU上的滤波器数据
    float *d_U,             // GPU输出U矩阵
    int num_filters,        // 滤波器数量（OC维度）
    const filter_shape_t fs,
    const U_shape_t us,
    const int64_t  collapsed_dim_size) {

    dim3 block(16, 16);
    dim3 grid(
        (collapsed_dim_size + block.x - 1) / block.x,
        (fs.w + block.y - 1) / block.y,
        num_filters * fs.h // 批处理维度（OC*H）
    );

    batch_filter_transform_kernel<<<grid, block>>>(
        d_packed_filters, d_U, num_filters, fs, us,collapsed_dim_size);
    cudaDeviceSynchronize();
}

// 在矩阵乘法sgemm完成后批量处理输出
void batch_output_transform(
    float *d_M,           // GPU上的中间结果M
    float *d_Y,            // GPU输出Y
    int batch_size,        // 批处理大小
    const tiling_info_t ti,
    const out_shape_t os,
    const int64_t  collapsed_dim_size)
{

    dim3 block(16, 16);
    dim3 grid(
        (collapsed_dim_size + block.x - 1) / block.x,
        (ti.tile_out_w + block.y - 1) / block.y,
        batch_size * ti.tile_out_h // 批处理维度
    );

    batch_output_transform_kernel<<<grid, block>>>(
        d_M, d_Y, batch_size, ti, os,collapsed_dim_size);
    cudaDeviceSynchronize();
}

// 调用配置
void launch_image_transform(float *d_input, float *d_V, int batch_size, tiling_info_t ti, int ic) {
    dim3 block(256);  // 每个block 256线程
    int total_threads = ic * ti.tile_in_w * ti.tile_in_h;
    dim3 grid((total_threads + block.x - 1) / block.x, batch_size);

    batch_image_transform_kernel<<<grid, block>>>(d_input, d_V, batch_size, ti, {6,6}, ic);
}
