
import torch
import cv2
import numpy as np
from lightglue import LightGlue, ALIKED #can replace aliked by superpoint to see how superpoint works
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import matplotlib.pyplot as plt
import os

# 1. Setup Device (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# 2. Load Models
# Using ALIKED as the feature extractor (Proposed improvement over SuperPoint)
print("Loading ALIKED and LightGlue models...")
extractor = ALIKED(max_num_keypoints=2048).eval().to(device) #can replace aliked by superpoint to see how superpoint works

# Load LightGlue with weights trained for ALIKED
# Note: 'features' argument must match the extractor type ('aliked')
matcher = LightGlue(features='aliked').eval().to(device) #can replace aliked by superpoint to see how superpoint works

# 3. Load Input Images
print("Loading images...")
# Ensure these files exist in the current directory
if not os.path.exists('img1.jpg') or not os.path.exists('img2.jpg'):
    print("Error: img1.jpg or img2.jpg not found.")
    exit()

image0 = load_image('img1.jpg').to(device)
image1 = load_image('img2.jpg').to(device)

# 4. Feature Extraction (Frontend)
print("Extracting features with ALIKED...")
feats0 = extractor.extract(image0)
feats1 = extractor.extract(image1)

# 5. Feature Matching
print("Matching features with LightGlue...")
matches01 = matcher({'image0': feats0, 'image1': feats1})

# 6. Data Post-processing
# Remove batch dimension to get raw keypoints and matches
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']

# Filter out the matched keypoints
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

print(f" Found {len(m_kpts0)} matches.")

# Geometric Verification & Pose Estimation
# Convert tensors to NumPy arrays for OpenCV compatibility
pts0 = m_kpts0.cpu().numpy()
pts1 = m_kpts1.cpu().numpy()

# 1. Compute Essential Matrix using RANSAC to reject outliers
# This validates the geometric consistency of the ALIKED matches
E, mask = cv2.findEssentialMat(pts0, pts1, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# 2. Recover Camera Pose (Rotation R, Translation t)
_, R, t, _ = cv2.recoverPose(E, pts0, pts1)

print("\n---------------------------------")
print("Visual Odometry Result:")
print(f"Rotation Matrix (R):\n{R}")
print(f"Translation Vector (t):\n{t}")
print("---------------------------------")

# 7. Visualization
print("Visualizing matches...")
axes = viz2d.plot_images([image0, image1])
viz2d.plot_matches(m_kpts0, m_kpts1, color='lime', lw=0.2)
plt.show()


