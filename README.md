# 🧠 Medical Image Classification Web App

A Streamlit-based web application for classifying oral medical conditions using a deep learning model trained with YOLO (Ultralytics). Users can upload an image and receive predicted labels with confidence scores in real time.

# 🚀 Features
📤 Upload images (.jpg, .jpeg, .png)

🧠 AI-powered classification using a trained .pt model

📊 Confidence score display

🎨 Clean and responsive UI with custom styling

⚡ Fast inference with preloaded model

🧩 Supports both classification and detection-based YOLO models

```
🏗️ Project Structure
.
├── streamlit_app.py     # Main Streamlit application
├── models.py            # Model loading and inference logic
├── exp-3.pt             # Trained YOLO model weights
├── img.jpg              # Background image (optional)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```
# 🧠 Model Details
Framework: YOLO (Ultralytics)

Format: .pt (PyTorch)

Task Type : Classification

Supported Classes : Canker Sores, Cold Sores, Mouth Cancer, Normal, Oral Thrush

# ⚙️ Installation
1. Clone the repository
```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Run the app
```
streamlit run streamlit_app.py
```

# 📦 Requirements
```
streamlit
ultralytics
torch
torchvision
numpy
pillow
opencv-python-headless
```

# ☁️ Deployment (Streamlit Cloud)
Push your project to GitHub

Go to Streamlit Cloud

Create a new app and link your repository

Ensure:

requirements.txt is included

.pt model file is present in the repo

Deploy and reboot if necessary

# ⚠️ Notes & Limitations
This application is intended for educational and experimental purposes only

Predictions may not be medically accurate

Model performance depends on:

training data quality

class balance

The "Normal" class may introduce bias in predictions

# 🧩 Future Improvements
🧠 Add Grad-CAM / explainability

🖼️ Display segmentation overlays

📱 Improve mobile responsiveness

🔍 Add confidence threshold warnings

📊 Support multiple models

# 📄 License
This project is open-source and available under the MIT License.

# 🙌 Acknowledgements
Ultralytics YOLO framework

Streamlit for rapid web deployment

Open-source datasets and tools used during development
