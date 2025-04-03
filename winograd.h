#pragma once

#ifndef WINOGRAD_H  // 添加头文件保护
#define WINOGRAD_H

#include "utils.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/*typedef struct image_shape_t image_shape_t;
typedef struct filter_shape_t filter_shape_t;
typedef struct tiling_info_t tiling_info_t;
typedef struct V_shape_t V_shape_t;
typedef struct U_shape_t U_shape_t;
typedef struct out_shape_t out_shape_t;*/


void winograd_convolution_gpu(
    float *d_image,
    float *d_filter,
    float *d_output,
    const image_shape_t is,
    const filter_shape_t fs,
    const tiling_info_t ti);

void launch_image_transform(float *d_input, float *d_V, int batch_size, tiling_info_t ti, int ic);

void batch_image_transform(
    float *d_packed_images, // GPU上的输入数据 (NHWC布局)
    float *d_V,            // GPU输出V矩阵
    int batch_size,        // 批处理大小
    const tiling_info_t ti,
    const V_shape_t vs,
    const int64_t  collapsed_dim_size);

void batch_filter_transform(
    float *d_packed_filters, // GPU上的滤波器数据
    float *d_U,             // GPU输出U矩阵
    int num_filters,        // 滤波器数量（OC维度）
    const filter_shape_t fs,
    const U_shape_t us,
    const int64_t  collapsed_dim_size);

void batch_output_transform(
    float *d_M,           // GPU上的中间结果M
    float *d_Y,            // GPU输出Y
    int batch_size,        // 批处理大小
    const tiling_info_t ti,
    const out_shape_t os,
    const int64_t  collapsed_dim_size);

/*typedef struct {
    int bs;   // batch size
    int h;     // image height
    int w;     // image width
    int ic;    // input channels
} image_shape_t;

typedef struct {
    int oc;    // output channels
    int ic;    // input channels
    int kh;    // kernel height
    int kw;    // kernel width
} filter_shape_t;

typedef struct {
    int tile_w;         // 横向分块数
    int tile_h;         // 纵向分块数
    int tile_in_w;      // 输入块宽度
    int tile_in_h;      // 输入块高度
    int tile_out_w;     // 输出块宽度
    int tile_out_h;     // 输出块高度
    int num_tiles;      // 总块数
    int tiles_per_row;   // 每行块数
    int input_w;        // 原始输入宽度
    int input_h;        // 原始输入高度
    int input_c;        // 原始输入通道数
} tiling_info_t;*/


tiling_info_t get_tiling_info(image_shape_t is, filter_shape_t fs);

#ifdef __cplusplus
}
#endif

#endif
