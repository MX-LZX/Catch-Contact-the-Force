# Catch-Contact-the-Force
A Multi-Contact-Point, Multi-Axis Foot-End Force Sensor for Legged Robots Magnetic-Hall Sensing via Spring Deformation

## How to use:
Repository Structure

physics/: Contains multiple physics-informed constraints for the PINN module

models/: Stores MLP model definitions and parameters

data_utils/: Holds raw data (12-channel Hall signals + contact point (x,y) + contact forces (fx, fy, fz)) and preprocessing scripts

data/: Contains processed datasets

checkpoints/: Stores pre-trained minimal models and loss curve figures

Setup

Install the required Python environment and dependencies listed in the project.

### Inference Only：

To predict directly from raw data:

```python evaluate.py```


To validate with the real sensor via serial port:

```python serial_inference.py --port COMxxx --baud 115200 --print-hz 25 --csv pred.csv```

### Training Your Own Model：

Use a 3-axis force/torque sensor as the ground truth to calibrate your CtF foot-end sensor.

Collect synchronized datasets: 12-channel Hall signals + contact point (x,y) + 3-axis forces (fx, fy, fz).

Insert the data into data_utils/data.xlsx.

Run preprocessing scripts:

```python Add_time_sequence.py```
```python convert_excel_dataset.py```


These will apply time-series formatting and generate .npy datasets.

Return to the root directory and start training:

```python train.py```


⚠️ Important: Make sure to back up your best_ema.pth file beforehand—each new training run will overwrite it.

## Notes

Hardware used for training: V100 16GB / RTX 3080 10GB

Training time: ~10 minutes for an initial dataset of ~20M samples

## License
Code is licensed under the **PolyForm Noncommercial License 1.0.0**.
Noncommercial use only; commercial use requires a separate license from the author.

## Attribution
Please retain the copyright notice and cite:

Meng, X., Lu, J., Ma, W. (2025). CtF: Multi-Contact, Multi-Axis Foot-End Force Estimation (PINN+MLP).

See `CITATION.cff` for a citation entry.
