import os
from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Class names for output
class_names = ['No DR (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)', 'Proliferative (4)']

# ==============================================
# ✅ Load Model (ResNet50 + Dropout)
# ==============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_features, 5)
)

# model_path = "D:\diabeticretinopathy\model-training\best_model_resnet50_new.pth"
model_path = r"D:\diabeticretinopathy\model-training\best_model_resnet50_new.pth"

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ==============================================
# ✅ Transform (same as during training)
# ==============================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================================
# ✅ Prediction Function
# ==============================================
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
    
    pred_class = class_names[pred.item()]
    return pred_class

# ==============================================
# ✅ Routes
# ==============================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def upload_and_predict():
    if "file" not in request.files:
        return "No file uploaded!"

    file = request.files["file"]
    if file.filename == "":
        return "No selected file!"

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    prediction = predict_image(file_path)

    return render_template("result.html", 
                           image_path=file_path, 
                           prediction=prediction)

# ==============================================
# ✅ Run App
# ==============================================
if __name__ == "__main__":
    app.run(debug=True)
