# Monocular Depth Estimation on KITTI

## Vision Transformer notebook

Open [ViT_Depth_Estimation.ipynb](ViT_Depth_Estimation.ipynb) in Colab with a GPU or in Jupyter with PyTorch, Pillow, NumPy, Matplotlib and scikit-learn installed. Extract [KITTI Eigen Split for Monocular Depth Estimation](https://www.kaggle.com/datasets/aalniak/kitti-eigen-split-for-monocular-depth-estimation) into the notebook's working directory under `data/`, or set `DATA_ROOT` to the extracted directory. Run the notebook from top to bottom.

The model learns from scratch using 16×16 patches, four Transformer encoder blocks and a convolutional upsampling head. It outputs a 256×512 depth map in metres. Training runs for up to 40 epochs, stops after five consecutive epochs without a lower validation MAE, and saves and restores the best weights from `vit_depth_best.pt`. The notebook reports validation MAE and RMSE over pixels with ground-truth measurements.

### Baseline analysis

The original notebook indexed depth files by filename alone, even though frame names repeat across drives. It also split individual frames randomly, which could put neighbouring frames from the same drive into both sets. The ViT notebook pairs frames by drive and frame ID, holds out entire drives, decodes the 16-bit depth values in metres, and excludes missing measurements. The old loss values below are historical. To compare U-Net and ViT, train both on the same corrected split with matching preprocessing and metrics. The validation split is used for checkpoint selection, so report final performance on a separate test split if available.

### Related datasets

- [NYU Depth V2 on Kaggle](https://www.kaggle.com/soumikrakshit/nyu-depth-v2) provides aligned indoor RGB and depth. Use it for a separate indoor experiment with a loader for its format.
- [KITTI LiDAR Based 2D Depth Images on Kaggle](https://www.kaggle.com/ahmedfawzyelaraby/kitti-lidar-based-2d-depth-images) provides LiDAR-derived targets in the same driving domain. Confirm that you can match corresponding RGB frames and check for overlapping KITTI drives before using it for evaluation.
- [Virtual KITTI 2](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/) provides synthetic driving scenes with dense depth. It is closer to KITTI's road domain, though the data comes from a renderer and needs a separate loader.

## Original U-Net baseline

## 1. Project Overview

This project implements a Monocular Depth Estimation system using a lightweight U-Net architecture built with PyTorch. The model predicts dense depth maps from a single RGB image.

Monocular depth estimation is an important problem in computer vision and is widely used in:

- Autonomous driving
- Robotics navigation
- 3D scene reconstruction
- Augmented reality systems
- Advanced driver assistance systems

The system is trained on the KITTI Eigen Split dataset using GPU acceleration (CUDA).

---

## 2. GitHub Project Structure

project-root/
│
├── data/
├── notebooks/
├── models/
├── outputs/
├── README.md
├── train.py
├── dataset.py
├── model.py
└── requirements.txt

Explanation:

- data/ → Dataset files  
- notebooks/ → Colab or Jupyter notebooks  
- models/ → Saved trained weights  
- outputs/ → Prediction visualizations  
- train.py → Training loop  
- dataset.py → Custom Dataset class  
- model.py → U-Net model architecture  

---

## 3. Dataset

Dataset: KITTI Eigen Split for Monocular Depth Estimation  
Source: Kaggle  
The original notebook reported 3984 pairs using filename-only matching. The ViT notebook prints the corrected count when run.

### Dataset Structure

data/
 ├── 2011_09_26/
 │    ├── image_02/data/*.png
 │    ├── proj_depth/groundtruth/image_02/*.png

Input:
- RGB Image (3 channels)

Output:
- Ground Truth Depth Map (1 channel, 16-bit PNG format)

---

## 4. Problem Statement

Given a single RGB image, the objective is to predict a dense depth map representing pixel-wise distance from the camera.

Function mapping:

f: RGB Image → Depth Map

Input Shape: (3 × 256 × 512)  
Output Shape: (1 × 256 × 512)

This is treated as a regression problem.

---

## 5. Model Architecture

Model Used: Custom Lightweight U-Net

Architecture Flow:

Input Image (3x256x512)  
        ↓  
Encoder Block 1  
        ↓  
MaxPooling  
        ↓  
Encoder Block 2  
        ↓  
Upsampling  
        ↓  
Skip Connection  
        ↓  
Final Convolution  
        ↓  
Depth Map (1x256x512)

### Layer Details

| Layer | Operation | Output Channels |
|-------|-----------|----------------|
| Enc1  | Conv + ReLU | 64 |
| Pool  | MaxPool2D | - |
| Enc2  | Conv + ReLU | 128 |
| Dec2  | Upsample + Conv | 64 |
| Dec1  | Conv + Final Conv | 1 |

---

## 6. Data Preprocessing

Steps performed:

1. Matched RGB images with corresponding depth maps.
2. Resized images to 256 × 512.
3. Converted images to PyTorch tensors.
4. Split dataset:
   - 80% Training
   - 20% Validation

No data augmentation was applied in this version.

---

## 7. Training Configuration

| Parameter | Value |
|------------|--------|
| Epochs | 5 |
| Batch Size | 8 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | MSE |
| Device | CUDA |

## 8. Loss Function

### Mean Squared Error (MSE)

\[
MSE = \frac{1}{n} \sum (y_{true} - y_{pred})^2
\]

**Reason:**

- Depth estimation is a regression task.
- MSE penalizes large errors strongly.
- It ensures smoother convergence during training.

---

## 9. Optimizer

**Optimizer Used:** Adam  
**Learning Rate:** 0.0001  

Adam combines:

- Momentum (to accelerate convergence)
- Adaptive learning rates (to adjust step size per parameter)

This makes it stable and efficient for deep learning models.

---

## 10. Training Process

The training loop performs the following steps:

1. Forward pass  
2. Loss computation  
3. Backpropagation  
4. Parameter update using optimizer  

After each epoch:

- Validation is performed on unseen data.
- Validation loss is calculated.
- Model generalization performance is monitored.

---

## 11. Training Results

| Epoch | Train MSE | Validation MSE |
|--------|------------|----------------|
| 1 | 1,884,299 | 1,860,941 |
| 2 | 1,880,627 | 1,846,348 |
| 3 | 1,870,745 | 1,862,811 |
| 4 | 1,864,705 | 1,836,735 |
| 5 | 1,859,686 | 1,827,723 |

### Observation

- Training and validation loss gradually decrease.
- The model is learning meaningful depth representations.
- More epochs are required for better convergence.
- Slight fluctuations indicate need for normalization and better regularization.

---

## 12. Prediction Pipeline

### Steps:

1. Load trained model  
2. Resize input image  
3. Convert image to tensor  
4. Perform forward pass  
5. Generate predicted depth map  

## 13. Limitations

- Only 5 epochs were used, which may lead to underfitting.
- No depth normalization was applied before training.
- No data augmentation techniques were used.
- A lightweight U-Net architecture was implemented instead of the full U-Net model.
- Only Mean Squared Error (MSE) loss was used.
- No advanced evaluation metrics were computed such as:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - Delta accuracy metrics

---

## 14. Future Improvements

- Train the model for 30–50 epochs for better convergence.
- Normalize depth values before training to improve numerical stability.
- Use advanced loss functions:
  - L1 Loss
  - BerHu Loss
  - Scale-Invariant Loss
- Apply data augmentation techniques such as random flip, crop, and brightness adjustment.
- Use a pretrained encoder (e.g., ResNet backbone) for better feature extraction.
- Add evaluation metrics such as:
  - RMSE
  - MAE
  - Delta accuracy
- Implement learning rate scheduling.
- Perform hyperparameter tuning.
- Use mixed precision training for faster performance and reduced memory usage.


![1](https://github.com/user-attachments/assets/f93bb56e-fa6f-42f1-96fa-ef4516cf2cf9)
![2](https://github.com/user-attachments/assets/baeb4346-c01d-444c-bd4e-ae09ba2eb955)
![3](https://github.com/user-attachments/assets/5234deb6-7f6d-4169-bcfc-d4178f1908f7)
![4](https://github.com/user-attachments/assets/4d5f6f3d-fc35-4043-ab47-47e55852c9a4)
![5](https://github.com/user-attachments/assets/35903daa-ea0b-4a82-a4d6-03223576256f)
![6](https://github.com/user-attachments/assets/efd53f61-27b2-47ad-bdf6-73875631f226)
