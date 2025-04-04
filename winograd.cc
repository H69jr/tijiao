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

// 图像打包张量结构体
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

// 滤波器张量结构体
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


void image_transform(float *__restrict__ packed_image,
                     float *__restrict__ V,
                     const V_shape_t vs,
                     const tiling_info_t ti,
                     const int64_t collapsed_dim_size) {
  typedef float(*packed_image_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  typedef float(*V_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  packed_image_tensor_t packed_image_tensor = (packed_image_tensor_t)packed_image;
  V_tensor_t V_tensor = (V_tensor_t)V;

  float z0, z1, z2, z3, z4, z5, z6;

  #pragma omp parallel for
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
    for (int64_t w = 0; w < ti.tile_in_w; ++w) {
      z6 = packed_image_tensor[0][w][idx];

      z0 = 4.0f * z6;

      z6 = packed_image_tensor[1][w][idx];

      z1 = -4.0f * z6;
      z2 = 4.0f * z6;
      z3 = -2.0f * z6;
      z4 = 2.0f * z6;
      z5 = 4.0f * z6;

      z6 = packed_image_tensor[2][w][idx];

      z0 += -5.0f * z6;
      z1 += -4.0f * z6;
      z2 += -4.0f * z6;
      z3 += -z6;
      z4 += -z6;

      z6 = packed_image_tensor[3][w][idx];

      z1 += z6;
      z2 += -z6;
      z3 += 2.0f * z6;
      z4 += -2.0f * z6;
      z5 += -5.0f * z6;

      z6 = packed_image_tensor[4][w][idx];

      z0 += z6;
      z1 += z6;
      z2 += z6;
      z3 += z6;
      z4 += z6;

      z6 = packed_image_tensor[5][w][idx];

      z5 += z6;

      V_tensor[0][w][idx] = z0;
      V_tensor[1][w][idx] = z1;
      V_tensor[2][w][idx] = z2;
      V_tensor[3][w][idx] = z3;
      V_tensor[4][w][idx] = z4;
      V_tensor[5][w][idx] = z5;
    }

    for (int64_t h = 0; h < ti.tile_in_h; ++h) {
      z6 = V_tensor[h][0][idx];

      z0 = 4.0f * z6;

      z6 = V_tensor[h][1][idx];

      z1 = -4.0f * z6;
      z2 = 4.0f * z6;
      z3 = -2.0f * z6;
      z4 = 2.0f * z6;
      z5 = 4.0f * z6;

      z6 = V_tensor[h][2][idx];

      z0 += -5.0f * z6;
      z1 += -4.0f * z6;
      z2 += -4.0f * z6;
      z3 += -z6;
      z4 += -z6;

      z6 = V_tensor[h][3][idx];

      z1 += z6;
      z2 += -z6;
      z3 += 2.0f * z6;
      z4 += -2.0f * z6;
      z5 += -5.0f * z6;

      z6 = V_tensor[h][4][idx];

      z0 += z6;
      z1 += z6;
      z2 += z6;
      z3 += z6;
      z4 += z6;

      z6 = V_tensor[h][5][idx];

      z5 += z6;

      V_tensor[h][0][idx] = z0;
      V_tensor[h][1][idx] = z1;
      V_tensor[h][2][idx] = z2;
      V_tensor[h][3][idx] = z3;
      V_tensor[h][4][idx] = z4;
      V_tensor[h][5][idx] = z5;
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
                            const tiling_info_t ti) {
  typedef float(*Y_tensor_t)[ti.tile_in_w][os.oc][ti.num_tiles];
  typedef float(*out_tensor_t)[os.oc][os.h][os.w];
  Y_tensor_t Y_tensor = (Y_tensor_t)Y;
  out_tensor_t out_tensor = (out_tensor_t)out;

  for (int64_t h = 0; h < ti.tile_out_h; ++h) {
    for (int64_t w = 0; w < ti.tile_out_w; ++w) {
      for (int64_t oc = 0; oc < os.oc; oc++) {
        for (int64_t tile = 0; tile < ti.num_tiles; tile++) {
          tile_index_t tidx = get_tile_index(tile, ti);
          int64_t batch = tidx.b, ww = tidx.tw, hh = tidx.th;
          if (hh * 4 + h < os.h && ww * 4 + w < os.w)
            out_tensor[batch][oc][(hh * 4 + h)][(ww * 4 + w)] = Y_tensor[h][w][oc][tile];
        }
      }
    }
  }
}

void sgemm(const int64_t M, const int64_t N, const int64_t K, float *A, float *B, float *C) {
  typedef float(*A_tensor_t)[K];
  typedef float(*B_tensor_t)[K];
  typedef float(*C_tensor_t)[M];
  A_tensor_t A_tensor = (A_tensor_t)A;
  B_tensor_t B_tensor = (B_tensor_t)B;
  C_tensor_t C_tensor = (C_tensor_t)C;

  #pragma omp parallel for
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      C_tensor[n][m] = 0;
      for (int64_t k = 0; k < K; ++k) {
        C_tensor[n][m] += A_tensor[m][k] * B_tensor[n][k];
      }
    }
  }
}

void winograd_convolution(
    float *__restrict__ image,  
    const int image_height,
    const int image_width,
    const int input_channel_num,
    float *__restrict__ filter, 
    const int output_channel_num,
    const int batch_num,
    float *__restrict__ out) {
  /* new vars of shape */
  const image_shape_t is = {.bs = batch_num, .ic = input_channel_num, .h = image_height, .w = image_width};
  const filter_shape_t fs = {.oc = output_channel_num, .ic = input_channel_num, .h = FLT_H, .w = FLT_W};
  const out_shape_t os = get_output_shape(is, fs);
  const tiling_info_t ti = get_tiling_info(is, os);
  const U_shape_t us = get_U_shape(fs, ti);
  const V_shape_t vs = get_V_shape(is, ti);

  float *packed_filter = (float *)malloc(sizeof(float) * fs.h * fs.w * fs.oc * fs.ic);
  float *packed_image = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic);
  float *U = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * us.ic);
  float *V = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic);
  float *M = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * vs.num_tiles);
  float *Y = (float *)malloc(sizeof(float) * ti.tile_out_h * ti.tile_in_w * os.oc * ti.num_tiles);

  filter_packing(filter, packed_filter, fs);
  filter_transform(packed_filter, U, fs, us, us.oc * us.ic);

  image_packing(image, packed_image, is, ti);
  image_transform(packed_image, V, vs, ti, vs.ic * vs.num_tiles);

  for (int64_t h = 0; h < ti.tile_in_h; ++h) {
    for (int64_t w = 0; w < ti.tile_in_w; ++w) {
      typedef float(*U_tensor_t)[ti.tile_in_w][us.oc][us.ic];
      typedef float(*V_tensor_t)[ti.tile_in_w][vs.num_tiles][vs.ic];
      typedef float(*M_tensor_t)[ti.tile_in_w][us.oc][vs.num_tiles];
      U_tensor_t U_tensor = (U_tensor_t)U;
      V_tensor_t V_tensor = (V_tensor_t)V;
      M_tensor_t M_tensor = (M_tensor_t)M;
      sgemm(us.oc,
            vs.num_tiles,
            us.ic,
            (float *)(U_tensor[h][w]),
            (float *)(V_tensor[h][w]),
            (float *)(M_tensor[h][w]));
      }
  }

  output_transform(M, Y, ti, us.oc * vs.num_tiles);
  output_unpacking_store(Y, out, os, ti);

  free(packed_filter);
  free(packed_image);
  free(U);
  free(V);
  free(M);
  free(Y);
}

