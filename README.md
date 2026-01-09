

# 👁️ DRInsight: AI-Powered Diabetic Retinopathy Screening

**DRInsight** is an automated diagnostic tool designed to detect and classify **Diabetic Retinopathy (DR)** from retinal fundus images. Developed with a focus on accessibility, this project aims to bridge the gap in eye care for regions with limited access to specialists, particularly in rural India.

---

## 📌 Project Overview
Diabetic Retinopathy is a leading cause of blindness globally. Early detection is critical for preventing permanent vision loss. DRInsight leverages Deep Learning to provide a rapid, affordable, and accurate screening tool deployed via a simple web interface.

* **Dataset:** EyePacs (via Kaggle) – 22 GB of retinal fundus images.
* **Model Architecture:** ResNet50 (Transfer Learning).
* **Deployment:** Flask-based Full-Stack Web Application.
* **Classification:** 5-stage clinical DR Grading.

---

## ⚙️ Technical Workflow

### 1. Data Preprocessing & Augmentation
Retinal images vary significantly in lighting and quality. To ensure model robustness, the following pipeline was implemented:
* **Resizing & Normalization:** Standardizing images for the ResNet50 input layer.
* **Color Distribution:** Correcting RGB channel variations to ensure consistent feature extraction.
* **Augmentation:** Applied rotation, flipping, and brightness adjustments to combat class imbalance and improve generalization.



### 2. Deep Learning Model: ResNet50
We utilized the **ResNet50 (Residual Network)** architecture, known for its ability to train deep networks effectively using **Skip Connections**.
* **Input:** Preprocessed $224 \times 224$ retinal images.
* **Mechanism:** Identifies subtle pathological features like microaneurysms, hemorrhages, and hard exudates.
* **Output:** Softmax probability across 5 severity classes.



### 3. Clinical Classification Stages
The model classifies images into five distinct categories based on international clinical standards:

| Stage | Classification | Description |
| :--- | :--- | :--- |
| **0** | **No DR** | No clinical signs of retinopathy. |
| **1** | **Mild NPDR** | Microaneurysms only. |
| **2** | **Moderate NPDR** | More than just microaneurysms but less than severe. |
| **3** | **Severe NPDR** | Significant hemorrhages and venous beading. |
| **4** | **PDR** | Neovascularization and vitreous hemorrhage. |

---

## 💻 Web Interface & Deployment
The system is deployed as a full-stack application designed for ease of use by healthcare workers in the field.

* **Backend:** Flask framework serving the trained TensorFlow model.
* **Frontend:** Clean, responsive UI for image uploads and result display.
* **User Journey:** 1. Upload a digital retinal scan.
    2. Model processes the image in real-time.
    3. UI displays the detected stage and medical recommendation.

---

## 🛠️ Tech Stack
* **Language:** Python 3.8+
* **Frameworks:** TensorFlow, Keras, Flask
* **Libraries:** OpenCV, NumPy, Pandas, Matplotlib

---
---
## 🚀 How to Run

### 1. Prerequisites
Ensure you have the following libraries installed:
```bash
pip install tensorflow flask opencv-python numpy pandas


---

# Clone the Repository
git clone [https://github.com/Saran-067/Diabetic-retinopathy-preprocessing-resnet50.git](https://github.com/Saran-067/Diabetic-retinopathy-preprocessing-resnet50.git)

# Navigate to the directory
cd Diabetic-retinopathy-preprocessing-resnet50

# Install dependencies
pip install -r requirements.txt

# Launch the application
python app.py
