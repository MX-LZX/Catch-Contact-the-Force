# Catch-Contact-the-Force
A Multi-Contact-Point, Multi-Axis Foot-End Force Sensor for Legged Robots Magnetic-Hall Sensing via Spring Deformation

<p align = "center">
<img src="Pic/Abstract.jpg" width = "430" height = "400" border="5" />
</p>

## Usage Instructions

I sincerely hope the CtF Foot-End Multi-Dimensional Force Sensor can reach a wide audience and help anyone in need of high-precision foot-ground contact detection or multi-axis force sensing—without being limited by the high cost of traditional strain-gauge-based sensors.

This repository includes:

Hardware: 3D models, titanium spring machining parameters, and material specifications;

Embedded PCB: Based on RP2040 + MLX90393, designed in LCSC EDA (an Altium Designer version will be provided in future updates);

Embedded Firmware: For sensor data acquisition and transmission;

PINN + MLP Algorithm Code: Runs on a personal PC for multi-axis force and contact-point estimation.

## How to Use

Hardware and PCB design files are provided in their respective folders;

Embedded firmware can be directly flashed to the RP2040;

Instructions for running the PINN + MLP algorithm are provided in a dedicated documentation file.

If you have any questions, feel free to contact me at: 936915881mxlzy@gmail.com

## PCB
<p align = "center">
<img src="Pic/PCB.png" width = "430" height = "260" border="5" />
</p>

## Structure
<p align = "center">
<img src="Pic/Structure.png" width = "430" height = "260" border="5" />
</p>

## License
Code is licensed under the **PolyForm Noncommercial License 1.0.0**.
Noncommercial use only; commercial use requires a separate license from the author.

## Attribution
Please retain the copyright notice and cite:

Meng, X., Lu, J., Ma, W. (2025). CtF: Multi-Contact, Multi-Axis Foot-End Force Estimation (PINN+MLP).

See `CITATION.cff` for a citation entry.
