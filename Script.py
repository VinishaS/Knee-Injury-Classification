import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


class MRNetDataset(Dataset):
    def __init__(self, data_dir, csv_path, plane, transform=None):
        self.data_dir = data_dir
        self.plane = plane
        self.transform = transform
        self.labels = pd.read_csv(csv_path, header=None, names=["scan_id", "label"])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        scan_id = str(self.labels.iloc[idx]["scan_id"]).zfill(4)
        label = self.labels.iloc[idx]["label"]
        image_path = os.path.join(self.data_dir, self.plane, f"{scan_id}.npy")
        image = np.load(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label

from torchvision import transforms

class SelectMiddleSlice:
    def __call__(self, img):
        return img[img.shape[0] // 2]

transform = transforms.Compose([
    SelectMiddleSlice(),
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
])

from torchvision import models
import torch.nn as nn

def get_resnet_model(num_classes=1):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

import torch
import torch.optim as optim
import torch.nn as nn

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-4, device="cuda"):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.view(-1), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

    return model

def extract_features(model, dataloader, device="cuda"):
    model.eval()
    features = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())

    return np.vstack(features)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_fusion_model(axial_features, coronal_features, sagittal_features, labels):
    combined_features = np.hstack([axial_features, coronal_features, sagittal_features])
    fusion_model = LogisticRegression(max_iter=1000)
    fusion_model.fit(combined_features, labels)
    return fusion_model


def evaluate_model(features, labels, logistic_regression):
    predicted_labels = logistic_regression.predict(features)
    predicted_probabilities = logistic_regression.predict_proba(features)
    if labels.ndim == 2:
        labels = np.argmax(labels, axis=1)
    if predicted_probabilities.shape[1] == 2:
        predicted_probabilities = predicted_probabilities[:, 1]
    accuracy = accuracy_score(labels, predicted_labels)
    precision = precision_score(labels, predicted_labels, average="weighted")
    recall = recall_score(labels, predicted_labels, average="weighted")
    f1 = f1_score(labels, predicted_labels, average="weighted")
    if predicted_probabilities.ndim == 1:
        roc_auc = roc_auc_score(labels, predicted_probabilities)
    else:
        roc_auc = roc_auc_score(labels, predicted_probabilities, multi_class="ovr")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {roc_auc:.4f}")


train_data_dir = "/Users/krishnannarayanan/Desktop/NEU/FALL 24/SML/MRNet-v1.0/SML Project/train"
valid_data_dir = "/Users/krishnannarayanan/Desktop/NEU/FALL 24/SML/MRNet-v1.0/SML Project/valid"
train_csv = {"abnormal": "train-abnormal.csv", "acl": "train-acl.csv", "meniscus": "train-meniscus.csv"}
valid_csv = {"abnormal": "valid-abnormal.csv", "acl": "valid-acl.csv", "meniscus": "valid-meniscus.csv"}

axial_train_dataset = MRNetDataset(train_data_dir, train_csv["abnormal"], "axial", transform)
axial_train_loader = DataLoader(axial_train_dataset, batch_size=16, shuffle=True)
axial_val_dataset = MRNetDataset(valid_data_dir, valid_csv["abnormal"], "axial", transform)
axial_val_loader = DataLoader(axial_val_dataset, batch_size=16, shuffle=False)

coronal_train_dataset = MRNetDataset(train_data_dir, train_csv["acl"], "coronal", transform)
coronal_train_loader = DataLoader(coronal_train_dataset, batch_size=16, shuffle=True)
coronal_val_dataset = MRNetDataset(valid_data_dir, valid_csv["acl"], "coronal", transform)
coronal_val_loader = DataLoader(coronal_val_dataset, batch_size=16, shuffle=False)

sagittal_train_dataset = MRNetDataset(train_data_dir, train_csv["meniscus"], "sagittal", transform)
sagittal_train_loader = DataLoader(sagittal_train_dataset, batch_size=16, shuffle=True)
sagittal_val_dataset = MRNetDataset(valid_data_dir, valid_csv["meniscus"], "sagittal", transform)
sagittal_val_loader = DataLoader(sagittal_val_dataset, batch_size=16, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
axial_model = train_model(get_resnet_model(), axial_train_loader, axial_val_loader, device=device)
coronal_model = train_model(get_resnet_model(), coronal_train_loader, coronal_val_loader, device=device)
sagittal_model = train_model(get_resnet_model(), sagittal_train_loader, sagittal_val_loader, device=device)
axial_features = extract_features(axial_model, axial_val_loader, device)
coronal_features = extract_features(coronal_model, coronal_val_loader, device)
sagittal_features = extract_features(sagittal_model, sagittal_val_loader, device)
combined_features = np.hstack([axial_features, coronal_features, sagittal_features])
labels = np.array([label for _, label in axial_val_loader.dataset])
fusion_model = train_fusion_model(axial_features, coronal_features, sagittal_features, labels)
evaluate_model(combined_features, labels, fusion_model)

