#include <Wire.h>
#include <LittleFS.h>
#include "Adafruit_MLX90393.h"

#include "MlxFilter.h"
#include "MlxTempCal.h"

// ============================== 硬件参数 ==============================
#define SDA_PIN       16
#define SCL_PIN       17
#define SERIAL_BAUD   115200

constexpr uint8_t SENSOR_ADDRS[4] = { 0x0C, 0x0D, 0x0E, 0x0F };
constexpr uint32_t SAMPLE_PERIOD_US = 25000UL;  // 25 ms
constexpr int      N_SENS = 4;

Adafruit_MLX90393 sensors[N_SENS];
MlxTempCal tempcal(sensors, N_SENS);

volatile bool g_streaming = true;      // 是否处于打印模式
MlxFilterMode g_mode = MLX_MODE_BALANCED;

uint32_t next_sample_us;
String cmd_buf;

// ============================== 归一化（新增） ==============================
// 零均值归一化：start 后先采 N 帧的均值作为基线，再输出 (x-mean)
static uint8_t  g_norm_window = 8;   // 建议 5~10, 默认 8
static bool     g_norm_enabled = true;
static bool     g_norm_ready = false;
static uint16_t g_norm_count = 0;
static float    g_norm_sum[4][3];     // [sensor][axis]
static float    g_norm_mean[4][3];

static inline void norm_reset() {
  memset(g_norm_sum, 0, sizeof(g_norm_sum));
  memset(g_norm_mean, 0, sizeof(g_norm_mean));
  g_norm_count = 0;
  g_norm_ready = false;
}

// 累计一帧（x/y/z 为 µT，已经做完温度偏置补偿）
static inline void norm_accumulate(const float x[4], const float y[4], const float z[4]) {
  for (int i=0;i<4;++i) {
    g_norm_sum[i][0] += x[i];
    g_norm_sum[i][1] += y[i];
    g_norm_sum[i][2] += z[i];
  }
  g_norm_count++;
  if (g_norm_count >= g_norm_window) {
    for (int i=0;i<4;++i) {
      g_norm_mean[i][0] = g_norm_sum[i][0] / g_norm_count;
      g_norm_mean[i][1] = g_norm_sum[i][1] / g_norm_count;
      g_norm_mean[i][2] = g_norm_sum[i][2] / g_norm_count;
    }
    g_norm_ready = true;
    Serial.println("NORM_READY");
  }
}

static inline void norm_apply_zero_mean(const float in[4], float out[4], int axis) {
  if (!g_norm_enabled || !g_norm_ready) {
    for (int i=0;i<4;++i) out[i] = in[i];
    return;
  }
  for (int i=0;i<4;++i) out[i] = in[i] - g_norm_mean[i][axis];
}

// ============================== 工具函数 ==============================
void applyFilterAll(MlxFilterMode mode) {
  MlxSetupAllFilters(sensors, N_SENS, mode);
}

// 读取单颗的片上温度（摄氏度）
float readTempC(int i) {
  int16_t tr;
  if (sensors[i].readTemperatureRaw(&tr)) {
    uint16_t u16 = (uint16_t)(((uint16_t)((uint8_t)((tr >> 8) & 0xFF)) << 8) |
                              (uint16_t)((uint8_t)(tr & 0xFF)));
    return mlxRawToCelsius(u16);
  }
  return NAN;
}

// ============================== 命令处理 ==============================
void handleSerialCommand() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r' || c == '\n') {
      cmd_buf.trim();
      if (cmd_buf.length() > 0) {

        // ---------- 流控 ----------
        if (cmd_buf.equalsIgnoreCase("start")) {
          g_streaming = true;
          norm_reset();                  // 每次 start 都重新做归一化基线
          Serial.println("Streaming: ON (collecting baseline...)");
        } else if (cmd_buf.equalsIgnoreCase("stop")) {
          g_streaming = false;
          Serial.println("Streaming: OFF");
        }

        // ---------- 滤波模式 ----------
        else if (cmd_buf.equalsIgnoreCase("mode_fast"))  { g_mode = MLX_MODE_FAST_RESP;  applyFilterAll(g_mode); Serial.println("Mode=FAST_RESP"); }
        else if (cmd_buf.equalsIgnoreCase("mode_bal"))   { g_mode = MLX_MODE_BALANCED;   applyFilterAll(g_mode); Serial.println("Mode=BALANCED"); }
        else if (cmd_buf.equalsIgnoreCase("mode_low"))   { g_mode = MLX_MODE_LOW_NOISE;  applyFilterAll(g_mode); Serial.println("Mode=LOW_NOISE"); }
        else if (cmd_buf.equalsIgnoreCase("mode_ultra")) { g_mode = MLX_MODE_ULTRA_LOW;  applyFilterAll(g_mode); Serial.println("Mode=ULTRA_LOW_NOISE"); }

        // ---------- 温标定 ----------
        else if (cmd_buf.equalsIgnoreCase("cal25") ||
                 cmd_buf.equalsIgnoreCase("cal40") ||
                 cmd_buf.equalsIgnoreCase("cal60")) {
          float Ttarget = cmd_buf.endsWith("25") ? 25.0f : cmd_buf.endsWith("40") ? 40.0f : 60.0f;
          for (int i=0;i<N_SENS;++i) {
            float mx,my,mz; tempcal.capturePoint(i, mx,my,mz, 128, 1000);
            if (Ttarget==25.0f) tempcal.setCal25(i, Ttarget, mx,my,mz);
            else if (Ttarget==40.0f) tempcal.setCal40(i, Ttarget, mx,my,mz);
            else tempcal.setCal60(i, Ttarget, mx,my,mz);
          }
          Serial.print("Captured cal @"); Serial.print(Ttarget); Serial.println("C");
        }
        else if (cmd_buf.equalsIgnoreCase("cal_fit")) {
          tempcal.fitAll(); Serial.println("Fitted c0/c1/c2 for all sensors.");
        }
        else if (cmd_buf.equalsIgnoreCase("cal_save")) {
          bool ok = tempcal.save("/mlx_cal.txt");
          Serial.println(ok ? "Saved /mlx_cal.txt" : "Save failed");
        }
        else if (cmd_buf.equalsIgnoreCase("cal_load")) {
          bool ok = tempcal.load("/mlx_cal.txt");
          Serial.println(ok ? "Loaded /mlx_cal.txt" : "Load failed");
        }
        else if (cmd_buf.equalsIgnoreCase("cal_stat")) {
          tempcal.printStatus();
        }

        // ---------- 温度查看 ----------
        else if (cmd_buf.equalsIgnoreCase("temp")) {
          char tbuf[160];
          float t[4]; for (int i=0;i<N_SENS;++i) t[i]=readTempC(i);
          int n=snprintf(tbuf,sizeof(tbuf),"TEMP_C: %.2f,%.2f,%.2f,%.2f\n",t[0],t[1],t[2],t[3]);
          Serial.write(tbuf,n);
        }

        // ---------- 归一化控制（新增） ----------
        else if (cmd_buf.equalsIgnoreCase("norm_on"))  { g_norm_enabled = true;  Serial.println("Normalization: ON"); }
        else if (cmd_buf.equalsIgnoreCase("norm_off")) { g_norm_enabled = false; Serial.println("Normalization: OFF"); }
        else if (cmd_buf.equalsIgnoreCase("norm_recal")) {
          norm_reset();
          Serial.println("Normalization: recalibrating baseline...");
        }
        else if (cmd_buf.startsWith("norm_window")) {
          // 形如：norm_window 8
          int space = cmd_buf.indexOf(' ');
          if (space > 0) {
            int val = cmd_buf.substring(space+1).toInt();
            if (val >= 5 && val <= 10) {
              g_norm_window = (uint8_t)val;
              norm_reset();
              Serial.print("Normalization window set to "); Serial.println(g_norm_window);
            } else {
              Serial.println("Normalization window must be 5..10");
            }
          } else {
            Serial.print("Current window="); Serial.println(g_norm_window);
          }
        }
        else if (cmd_buf.equalsIgnoreCase("norm_stat")) {
          Serial.print("norm_enabled="); Serial.print(g_norm_enabled ? "1":"0");
          Serial.print(", ready="); Serial.print(g_norm_ready ? "1":"0");
          Serial.print(", window="); Serial.println(g_norm_window);
          if (g_norm_ready) {
            for (int i=0;i<4;++i) {
              char l[96];
              int n = snprintf(l, sizeof(l), "BL[%d] mean(x,y,z)= %.4f, %.4f, %.4f\n",
                               i, g_norm_mean[i][0], g_norm_mean[i][1], g_norm_mean[i][2]);
              Serial.write(l, n);
            }
          }
        }

      }
      cmd_buf = "";
    } else {
      if (cmd_buf.length() < 64) cmd_buf += c;
    }
  }
}

// ============================== SETUP/LOOP ==============================
void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial);

  // 文件系统
  if (!LittleFS.begin()) {
    Serial.println("LittleFS mount failed");
  } else {
    if (tempcal.load("/mlx_cal.txt")) Serial.println("Loaded previous calibration.");
  }

  Wire.setSDA(SDA_PIN);
  Wire.setSCL(SCL_PIN);
  Wire.begin();
  Wire.setClock(1000000);  // 如总线较长/噪声大，可降到 400kHz

  // 传感器初始化
  for (int i=0;i<N_SENS;++i) {
    if (!sensors[i].begin_I2C(SENSOR_ADDRS[i], &Wire)) {
      Serial.print("Sensor "); Serial.print(i); Serial.println(" init failed!");
      while (1);
    }
    sensors[i].setResolution(MLX90393_X, MLX90393_RES_16);
    sensors[i].setResolution(MLX90393_Y, MLX90393_RES_16);
    sensors[i].setResolution(MLX90393_Z, MLX90393_RES_16);
    sensors[i].setGain(MLX90393_GAIN_1X);
  }

  // 默认滤波：BALANCED
  applyFilterAll(MLX_MODE_BALANCED);

  // 启动一次单次测量
  for (int i=0;i<N_SENS;++i) sensors[i].startSingleMeasurement();
  next_sample_us = micros() + SAMPLE_PERIOD_US;

  // 归一化初始化
  norm_reset();

  Serial.println("Ready. Type 'start' to stream.");
}

void loop() {
  handleSerialCommand();

  uint32_t now = micros();
  if ((int32_t)(now - next_sample_us) < 0) return;

  float x[N_SENS], y[N_SENS], z[N_SENS];
  for (int i=0;i<N_SENS;++i) sensors[i].readMeasurement(&x[i], &y[i], &z[i]);

  // 运行时温度偏置补偿
  for (int i=0;i<N_SENS;++i) {
    int16_t tr;
    if (sensors[i].readTemperatureRaw(&tr)) {
      uint16_t u16 = (uint16_t)(((uint16_t)((uint8_t)((tr >> 8) & 0xFF)) << 8) |
                                (uint16_t)((uint8_t)(tr & 0xFF)));
      float T = mlxRawToCelsius(u16);
      tempcal.applyBias(i, T, x[i], y[i], z[i]);
    }
  }

  // 启动下一帧测量（管线化）
  for (int i=0;i<N_SENS;++i) sensors[i].startSingleMeasurement();
  next_sample_us += SAMPLE_PERIOD_US;

  // ---- 归一化流程 ----
  if (g_streaming) {
    if (!g_norm_ready) {
      // 先采 N 帧做基线，不打印数据
      norm_accumulate(x, y, z);
      return;
    }

    // 基线已经就绪：输出零均值后的 12 维
    float xn[4], yn[4], zn[4];
    norm_apply_zero_mean(x, xn, 0);
    norm_apply_zero_mean(y, yn, 1);
    norm_apply_zero_mean(z, zn, 2);

    char buf[220];
    int n = snprintf(buf, sizeof(buf),
      "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
      xn[0], yn[0], zn[0],
      xn[1], yn[1], zn[1],
      xn[2], yn[2], zn[2],
      xn[3], yn[3], zn[3]);
    Serial.write(buf, n);
  }
}
