import os
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T

# 1. Define the transformations:
# Normalize scale to fit the math, and sample exactly 1024 points per object
pre_transform = T.NormalizeScale()
transform = T.SamplePoints(1024)

# 2. Tell PyG to download and process ModelNet10
print("Downloading and processing ModelNet10... (This might take a few minutes)")
dataset = ModelNet(
    root="./data/ModelNet10",
    name="10",
    train=True,
    transform=transform,
    pre_transform=pre_transform,
)

# 3. Verify the data
print("\n--- Dataset Ready! ---")
print(f"Total 3D objects downloaded: {len(dataset)}")
print(f"Number of categories to classify: {dataset.num_classes}")

# 4. Look at the very first 3D object
data = dataset[0]
print(f"\nFirst Object Data:")
print(f"Points (Nodes): {data.pos.size(0)}")
print(f"Coordinates (X,Y,Z features): {data.pos.size(1)}")
