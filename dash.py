import os
import numpy as np
import pandas as pd
import torch
from torchvision import transforms, models
import torch.nn as nn
import streamlit as st
from sklearn.linear_model import LogisticRegression

def get_resnet_model(num_classes=1):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
axial_model = get_resnet_model()
axial_model.load_state_dict(torch.load("axial_model.pth", map_location=device))
axial_model = axial_model.to(device)
axial_model.eval()
coronal_model = get_resnet_model()
coronal_model.load_state_dict(torch.load("coronal_model.pth", map_location=device))
coronal_model = coronal_model.to(device)
coronal_model.eval()
sagittal_model = get_resnet_model()
sagittal_model.load_state_dict(torch.load("sagittal_model.pth", map_location=device))
sagittal_model = sagittal_model.to(device)
sagittal_model.eval()
fusion_model = LogisticRegression(max_iter=1000)
fusion_model.coef_ = np.load("fusion_model_coef.npy")
fusion_model.intercept_ = np.load("fusion_model_intercept.npy")
fusion_model.classes_ = np.array([0, 1, 2])  # Ensure this matches your target classes (e.g., ACL Tear, Meniscus Tear, Abnormality)
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
def predict_injury(axial_image, coronal_image, sagittal_image, axial_model, coronal_model, sagittal_model, fusion_model):
    axial_features = axial_model(axial_image.to(device)).detach().cpu().numpy()
    coronal_features = coronal_model(coronal_image.to(device)).detach().cpu().numpy()
    sagittal_features = sagittal_model(sagittal_image.to(device)).detach().cpu().numpy()
    combined_features = np.hstack([axial_features, coronal_features, sagittal_features])
    prediction = fusion_model.predict(combined_features)[0]
    prediction_probabilities = fusion_model.predict_proba(combined_features)[0]
    return prediction, prediction_probabilities
def main():
    st.title("MRI Injury Detection Dashboard")
    st.write("Upload three MRI slices (Axial, Coronal, Sagittal) to predict the type of injury.")
    axial_file = st.file_uploader("Upload Axial Plane (.npy)", type=["npy"])
    coronal_file = st.file_uploader("Upload Coronal Plane (.npy)", type=["npy"])
    sagittal_file = st.file_uploader("Upload Sagittal Plane (.npy)", type=["npy"])

    if axial_file and coronal_file and sagittal_file:
        axial_image = torch.tensor(transform(np.load(axial_file))).unsqueeze(0).to(device)
        coronal_image = torch.tensor(transform(np.load(coronal_file))).unsqueeze(0).to(device)
        sagittal_image = torch.tensor(transform(np.load(sagittal_file))).unsqueeze(0).to(device)
        prediction, prediction_probabilities = predict_injury(
            axial_image, coronal_image, sagittal_image, axial_model, coronal_model, sagittal_model, fusion_model
        )
        class_map = {0: "ACL Tear", 1: "Meniscus Tear", 2: "Abnormality"}
        st.subheader("Prediction Results:")
        st.write(f"Predicted Injury: **{class_map[prediction]}**")
        st.subheader("Diagnosis Report:")
        if prediction == 0:
            st.write("ACL Tear detected. Consult an orthopedic specialist for further evaluation and potential surgery.")
        elif prediction == 1:
            st.write("Meniscus Tear detected. Rest and physical therapy may be required.")

if __name__ == "__main__":
    main()