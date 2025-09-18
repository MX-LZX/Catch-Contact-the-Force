#pragma once
#include "Adafruit_MLX90393.h"

enum MlxFilterMode {
  MLX_MODE_FAST_RESP = 0,  // OSR=1x, DF=1
  MLX_MODE_BALANCED  = 1,  // OSR=2x, DF=3
  MLX_MODE_LOW_NOISE = 2,  // OSR=4x, DF=5
  MLX_MODE_ULTRA_LOW = 3   // OSR=8x, DF=6
};

// 应用到单个设备
inline void MlxApplyFilterProfile(Adafruit_MLX90393& dev, MlxFilterMode mode) {
  mlx90393_oversampling_t osr = MLX90393_OSR_0;
  mlx90393_filter_t       df  = MLX90393_FILTER_1;

  switch (mode) {
    case MLX_MODE_FAST_RESP: osr = MLX90393_OSR_0; df = MLX90393_FILTER_1; break; // 1x, 1
    case MLX_MODE_BALANCED:  osr = MLX90393_OSR_1; df = MLX90393_FILTER_3; break; // 2x, 3
    case MLX_MODE_LOW_NOISE: osr = MLX90393_OSR_2; df = MLX90393_FILTER_5; break; // 4x, 5
    case MLX_MODE_ULTRA_LOW: osr = MLX90393_OSR_3; df = MLX90393_FILTER_6; break; // 8x, 6
  }
  dev.setOversampling(osr);
  dev.setFilter(df);
}

// 批量应用
inline void MlxSetupAllFilters(Adafruit_MLX90393 sensors[], int count, MlxFilterMode mode) {
  for (int i = 0; i < count; ++i) MlxApplyFilterProfile(sensors[i], mode);
}
