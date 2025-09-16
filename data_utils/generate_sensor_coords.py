import numpy as np
import os

def generate_sensor_coords(output_path='../data/sensor_coords.npy'):
    # 4个传感器固定坐标
    coords = np.array([
        [12.0,  0.0, 0.0],   # H1
        [-12.0, 0.0, 0.0],   # H2
        [0.0,  12.0, 0.0],   # H3
        [0.0, -12.0, 0.0],   # H4
    ], dtype=np.float32)

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, coords)
    print(f"✅ 已保存 sensor_coords.npy 到 {output_path}")
    print(coords)

if __name__ == '__main__':
    generate_sensor_coords()
