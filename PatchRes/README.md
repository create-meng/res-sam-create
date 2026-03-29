# TileRes Anomaly Detection

## Overview

Performing anomaly detection on GPR image data by TileRes. 

## Data Structure

The data for this project is organized as follows:

- **Images for Testing**:   `./data/images/`
  - Place the corresponding images for the test dataset in this directory.

- **Images for Training**:  `./data/normal/`
  - Place the normal images for training the anomaly detection model in this directory.

## Running the Demo

This project works well with **python>=3.8** (recommended **python=3.9**).

For Res-SAM **V4/V5** experiments, see `../experiments/`（`*_v4.py` / `*_v5.py`）and `../experiments/run_all.py`. V3 snapshot lives under `../archive/experiments_v3_snapshot_20260326/`.

Ensure you have the necessary dependencies installed. You can install them using the requirements.txt file:

``` bash
pip install -r requirements.txt
```

To run the TileRes anomaly detection, execute the following command:

```bash
python main.py
```

## Results
The results of the anomaly detection will be saved in the following directories:

- **Image with Anomaly Frame**:   ./output/frames/
  - This directory contains images with frames drawn around detected anomalies.
- **Anomaly Masks**:              ./output/masks/
  - This directory contains the generated anomaly masks.


