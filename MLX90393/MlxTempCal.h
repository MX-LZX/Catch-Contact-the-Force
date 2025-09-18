#pragma once
#include <Arduino.h>
#include <LittleFS.h>
#include "Adafruit_MLX90393.h"

// ============ 三点温标定 + 偏置补偿（不依赖寄存器访问） ============
//
// 用法概览：
//   MlxTempCal tempcal(sensors, 4);
//   // —— 标定流程（静磁环境，温度稳定到指定点后执行）
//   // 25°C:
//   float mx,my,mz; tempcal.capturePoint(0, mx,my,mz); tempcal.setCal25(0, 25.0f, mx,my,mz); // 每颗都要
//   // 40°C / 60°C 同理...
//   tempcal.fitAll();          // 三点拟合二次多项式
//   tempcal.save("/mlx_cal.txt");  // 掉电保存
//
//   // —— 上电：
//   if (LittleFS.begin()) tempcal.load("/mlx_cal.txt");
//
//   // —— 运行时：采样后按当前温度调用
//   tempcal.applyBias(i, T_celsius, x, y, z);
//
// 注：本头文件不再更改 TCMP_EN（片上温度补偿位）。
// ===================================================================

#ifndef MLX_CAL_SAVE_PATH
#define MLX_CAL_SAVE_PATH "/mlx_cal.txt"
#endif

// 可选：片上温度原始计数换算为摄氏度的典型系数（若你已有独立温度通道换算，可不使用）
static constexpr float MLX_T25_LSB   = 46244.0f;
static constexpr float MLX_LSB_PER_C = 45.2f;
inline float mlxRawToCelsius(uint16_t raw_u16) {
  return (raw_u16 - MLX_T25_LSB) / MLX_LSB_PER_C + 25.0f;
}

// 每颗传感器每轴：二次多项式偏置  offset(T) = c0 + c1*(T-25) + c2*(T-25)^2
struct MlxCalCoeff {
  bool have = false;
  float c0[3]{0,0,0}, c1[3]{0,0,0}, c2[3]{0,0,0};
};

// 三个温点的采样均值（静磁环境）
struct MlxCalPoint {
  bool have = false;
  float T = 0.0f;          // 实际记录的温度（建议 25/40/60）
  float mean[3]{0,0,0};    // X/Y/Z 平均值（单位同 readMeasurement 输出，通常 µT）
};

class MlxTempCal {
public:
  MlxTempCal(Adafruit_MLX90393* arr, int n) : devs(arr), ndev(n) {}

  // 采集某一温点的均值（静磁环境；frames 越大越稳）
  bool capturePoint(int sensor_idx, float& mx, float& my, float& mz,
                    int frames=128, int us_delay=1000) {
    if (sensor_idx < 0 || sensor_idx >= ndev) return false;
    float sx=0, sy=0, sz=0;
    for (int k=0;k<frames;++k) {
      float x,y,z;
      devs[sensor_idx].readMeasurement(&x,&y,&z);
      sx+=x; sy+=y; sz+=z;
      if (us_delay>0) delayMicroseconds(us_delay);
    }
    mx=sx/frames; my=sy/frames; mz=sz/frames;
    return true;
  }

  // 记录三点：25/40/60（T 可微调为你的实际稳定温度）
  void setCal25(int i, float T, float mx, float my, float mz) { setCal_(cal25, i, T, mx,my,mz); }
  void setCal40(int i, float T, float mx, float my, float mz) { setCal_(cal40, i, T, mx,my,mz); }
  void setCal60(int i, float T, float mx, float my, float mz) { setCal_(cal60, i, T, mx,my,mz); }

  // 用三点拟合每轴二次多项式 offset(T)（dT = T - 25）
  void fitAll() {
    for (int i=0;i<ndev;++i) {
      if (!(cal25[i].have && cal40[i].have && cal60[i].have)) { coef[i].have=false; continue; }
      float dT1=cal25[i].T-25.0f, dT2=cal40[i].T-25.0f, dT3=cal60[i].T-25.0f;
      for (int ax=0; ax<3; ++ax) {
        float y1=cal25[i].mean[ax], y2=cal40[i].mean[ax], y3=cal60[i].mean[ax];
        fit3_(dT1,y1, dT2,y2, dT3,y3, coef[i].c0[ax], coef[i].c1[ax], coef[i].c2[ax]);
      }
      coef[i].have = true;
    }
  }

  // 运行时：按当前温度补偿偏置（只改 x/y/z 的偏置项）
  inline void applyBias(int i, float T_celsius, float& x, float& y, float& z) const {
    if (i<0 || i>=ndev) return;
    if (!coef[i].have) return;
    float dT = T_celsius - 25.0f;
    x -= (coef[i].c0[0] + coef[i].c1[0]*dT + coef[i].c2[0]*dT*dT);
    y -= (coef[i].c0[1] + coef[i].c1[1]*dT + coef[i].c2[1]*dT*dT);
    z -= (coef[i].c0[2] + coef[i].c1[2]*dT + coef[i].c2[2]*dT*dT);
  }

  // 掉电保存（文本格式）
  bool save(const char* path=MLX_CAL_SAVE_PATH) {
    File f = LittleFS.open(path, "w");
    if (!f) return false;
    // 头
    f.println(F("MLXCAL v1"));
    // 每个传感器：一行状态，三行系数
    char line[96];
    for (int i=0;i<ndev;++i) {
      int have = coef[i].have ? 1 : 0;
      snprintf(line, sizeof(line), "S %d %d\n", i, have);
      f.print(line);
      for (int ax=0; ax<3; ++ax) {
        // c0 c1 c2
        snprintf(line, sizeof(line), "%.9f %.9f %.9f\n",
                 coef[i].c0[ax], coef[i].c1[ax], coef[i].c2[ax]);
        f.print(line);
      }
    }
    f.close();
    return true;
  }

  // 掉电加载
  bool load(const char* path=MLX_CAL_SAVE_PATH) {
    File f = LittleFS.open(path, "r");
    if (!f) return false;
    String header = f.readStringUntil('\n'); header.trim();
    if (!header.startsWith("MLXCAL")) { f.close(); return false; }

    for (int i=0;i<ndev;++i) {
      String line = f.readStringUntil('\n'); line.trim();
      // 解析 "S <idx> <have>"
      int idx=-1, have=0;
      if (sscanf(line.c_str(), "S %d %d", &idx, &have) != 2) { f.close(); return false; }
      if (idx != i) { f.close(); return false; }
      coef[i].have = (have != 0);

      // 三行：c0 c1 c2
      for (int ax=0; ax<3; ++ax) {
        line = f.readStringUntil('\n'); line.trim();
        double c0=0, c1=0, c2=0;
        if (sscanf(line.c_str(), "%lf %lf %lf", &c0, &c1, &c2) != 3) { f.close(); return false; }
        coef[i].c0[ax] = (float)c0;
        coef[i].c1[ax] = (float)c1;
        coef[i].c2[ax] = (float)c2;
      }
    }
    f.close();
    return true;
  }

  // 打印当前系数状态
  void printStatus(Stream& s=Serial) const {
    for (int i=0;i<ndev;++i) {
      s.print("S"); s.print(i); s.print(": ");
      if (!coef[i].have) { s.println("no coeff"); continue; }
      for (int ax=0; ax<3; ++ax) {
        s.printf("ax%d c0=%.6f c1=%.6f c2=%.6f  ",
                 ax, coef[i].c0[ax], coef[i].c1[ax], coef[i].c2[ax]);
      }
      s.println();
    }
  }

  // 公开成员（可直接读取）
  MlxCalCoeff  coef[4];
  MlxCalPoint  cal25[4], cal40[4], cal60[4];

private:
  // 三点拟合二次多项式（克拉默法则）
  static void fit3_(float dT1, float y1, float dT2, float y2, float dT3, float y3,
                    float& c0, float& c1, float& c2) {
    float x1=dT1, x2=dT2, x3=dT3;
    float a11=1, a12=x1, a13=x1*x1;
    float a21=1, a22=x2, a23=x2*x2;
    float a31=1, a32=x3, a33=x3*x3;
    float det = a11*(a22*a33-a23*a32) - a12*(a21*a33-a23*a31) + a13*(a21*a32-a22*a31);
    if (fabs(det) < 1e-9f) { c0=y1; c1=0; c2=0; return; }
    float det0 = y1*(a22*a33-a23*a32) - a12*(y2*a33-a23*y3) + a13*(y2*a32-a22*y3);
    float det1 = a11*(y2*a33-a23*y3) - y1*(a21*a33-a23*a31) + a13*(a21*y3-a31*y2);
    float det2 = a11*(a22*y3-a32*y2) - a12*(a21*y3-a31*y2) + y1*(a21*a32-a31*a22);
    c0 = det0/det; c1 = det1/det; c2 = det2/det;
  }

  static void setCal_(MlxCalPoint* arr, int i, float T, float mx, float my, float mz) {
    if (i < 0) return;
    arr[i].have = true; arr[i].T = T;
    arr[i].mean[0]=mx; arr[i].mean[1]=my; arr[i].mean[2]=mz;
  }

  Adafruit_MLX90393* devs;
  int ndev;
};
