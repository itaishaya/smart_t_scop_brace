\# Smart T-Scope: Section 8 Data Analysis \& ML Pipeline



\## Overview

This repository contains the data analysis and machine learning pipeline for the \*\*Smart T-Scope\*\*, an AI-based wearable system for monitoring knee movement during rehabilitation. 



The pipeline ingests raw inertial measurement unit (IMU) data from dual MPU6050 sensors mounted on the user's thigh and shank. It processes this data to estimate real-time anatomical knee joint angles, detects specific gait events, calculates the Range of Motion (ROM), and utilizes supervised machine learning to classify user activities (Standing, Sitting, Walking).



\## File Structure



\### 1. Source Code

\* `t\_scope\_ml\_pipeline.py`: The core Python script responsible for calibration, signal processing, feature extraction, model training, and data visualization.



\### 2. Calibration Datasets

These files establish the baseline orientation of the sensors relative to gravity to map the local sensor axes to an anatomical reference frame.

\* `cal\_measurement\_tscope\_standing.csv`: Used to define the vertical (Z) axis.

\* `cal\_measurement\_tscope\_on\_the\_floor\_box\_upwards.csv`: Used to define the anterior/normal axis.

\* `cal\_measurement\_tscope\_on\_the\_floor\_box\_sideway.csv`: Used to define the lateral (flexion/extension) axis.



\### 3. Training Datasets (Labeled Ground Truth)

These files contain isolated, specific movements used to train the Random Forest and XGBoost classifiers.

\* \*\*Walking Class:\*\* `walking.csv`

\* \*\*Sitting Class:\*\* `sitting\_straightening\_leg.csv`

\* \*\*Standing Class:\*\* `standing\_bending.csv`, `standing\_raising\_leg\_no\_bend.csv`



\### 4. Inference / Measurement Datasets

\* `activities.csv`: A continuous, mixed-activity session used as the unseen test set to evaluate the AI model's real-time classification capabilities.



\### 5. Outputs (Generated Automatically)

When the script is executed, it generates a `section8\_outputs` directory containing:

\* `/plots/`: High-resolution `.png` graphs illustrating raw signals, smoothed knee angles, gait detection peaks, and AI classification overlays.

\* `/tables/`: Extracted statistical data in `.csv` format, including calibration summaries, stride-by-stride ROM, and model predictions.



\---



\## Software Architecture and Data Analysis Pipeline



The Smart T-Scope relies on a robust Python-based software pipeline to process the continuous stream of raw Inertial Measurement Unit (IMU) data, translate it into clinically relevant kinematic metrics, and autonomously classify user activities. The pipeline is structured into six sequential stages:



\### 1. Data Ingestion and Preprocessing

Data is collected from two MPU6050 sensors via I²C communication and saved locally as CSV files. The lower sensor (S1, address 0x68) is mounted on the tibia, while the upper sensor (S2, address 0x69) is mounted on the femur. The script first normalizes the time vectors so that all measurements begin at `t = 0`. Because real-world I²C sampling can occasionally experience micro-delays or log identical timestamps, the ingestion function includes a robust fallback mechanism. It calculates the median time difference between samples to estimate the true sampling frequency (Fs) and interpolates missing timestamps to maintain a strictly continuous time-series.



\### 2. Sensor Calibration and Anatomical Mapping

Because the two IMUs are not physically mounted in the exact same orientation relative to the brace's chassis, their local coordinate systems must be mapped to a shared anatomical frame of reference. The software achieves this by reading three baseline static calibration files:

\* \*\*Standing Position:\*\* Determines which local axes align with the vertical gravity vector (e.g., establishing that the lower sensor's vertical axis corresponds to `-Ax`).

\* \*\*Horizontal Box-Up:\*\* Defines the anterior (forward-facing) normal axis.

\* \*\*Lateral Position:\*\* Identifies the specific gyroscope axis corresponding to knee flexion and extension.



By calculating the mean acceleration vectors in these known positions, the script programmatically assigns the raw data streams to their respective anatomical roles.



\### 3. Kinematic Processing and the Complementary Filter

Once the data is anatomically aligned, the system calculates the continuous Range of Motion (ROM) of the knee joint. To solve the issue of accelerometer noise during dynamic impacts and gyroscope angular drift over time, the pipeline implements a computationally lightweight \*\*Complementary Filter\*\*:

`θ = α \* (θ\_prev + ω \* Δt) + (1 - α) \* θ\_acc`



The filter relies heavily on the integrated gyroscope data for short-term, high-frequency motion tracking, while continuously applying a small correction weight from the absolute gravitational tilt angle measured by the accelerometer. The absolute angle of the thigh is then subtracted from the absolute angle of the shank to yield the relative, drift-free knee flexion angle. A moving average is subsequently applied to smooth the resulting waveform.



\### 4. Gait Segmentation and Feature Extraction

To analyze specific walking patterns, the continuous data stream must be segmented into discrete gait cycles (strides). The software calculates the "Absolute Jerk" (the derivative of acceleration over time) and applies a peak-detection algorithm. A heuristic threshold combined with a minimum temporal distance is used to identify the precise moment of Heel Strike. By defining a single stride as the data between two consecutive Heel Strikes, the software dynamically extracts key performance metrics per step, including the maximum flexion angle, minimum extension angle, and the total dynamic ROM.



\### 5. Supervised Machine Learning Integration

To transition the Smart T-Scope from a passive logging device to an active, intelligent monitoring system, the pipeline integrates a supervised machine learning architecture utilizing the `scikit-learn` and `xgboost` libraries. 

\* \*\*Training Phase:\*\* The system ingests several explicitly labeled datasets representing three core physical states: \*Standing\*, \*Sitting\*, and \*Walking\*. The script applies a 1.0-second rolling window across these datasets to extract engineered features: Mean Knee Angle, Standard Deviation of the Knee Angle, Mean Absolute Jerk, and the raw anterior acceleration.

\* \*\*Model Generation:\*\* These feature vectors are used to train two parallel algorithms: a \*\*Random Forest Classifier\*\* and an \*\*XGBoost Classifier\*\*. 

\* \*\*Inference Phase:\*\* Once trained, the models are deployed against continuous, unseen "mixed-activity" datasets. The software parses the live data using the same rolling window, passes the engineered features into the trained models, and outputs a predicted activity state for every millisecond of the recording. 



\### 6. Output Generation and Reporting

The final stage of the pipeline acts as an automated reporting tool. It exports the calculated ROM statistics, stride tables, and AI predictions into structured CSV files. Furthermore, using `matplotlib`, it generates a suite of high-resolution graphical plots, successfully overlaying the AI's predicted activity states (color-coded as Standing, Sitting, or Walking) directly onto the patient's continuous knee flexion waveform, providing a clear, visual summary of the rehabilitation session.



\---



\## Installation \& Setup



\### Prerequisites

Ensure you have Python 3.8+ installed. You will need the following Python packages:



```bash

pip install numpy pandas matplotlib scikit-learn xgboost





\## Execution

1. Clone this repository to your local machine.
2. Ensure all the .csv data files are in the same directory as t_scope_ml_pipeline.py.
3. Run the script:
```bash
python t_scope_ml_pipeline.py

