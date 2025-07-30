# MSI Generator

> High-fidelity multispectral image generation using a Swin-U-Net model. Converts RGB images into 6-channel multispectral outputs through an interactive UI and scalable API.

---

## 🚀 Features

- 🎯 **High-Fidelity Conversion**  
  Generate accurate 6-channel multispectral images using a Swin-U-Net model.

- 🖼️ **Interactive Web UI**  
  Built with Streamlit for easy image upload and output visualization.

- ⚙️ **Scalable FastAPI Backend**  
  Dockerized REST API serving the PyTorch model with CPU or GPU support.

- 🧩 **Large Image Support**  
  Automatically tiles large images into 224×224 patches and stitches outputs.

- 🧱 **Modular Codebase**  
  Cleanly separated modules for training, inference, and deployment.

---

## 🏗️ Architecture Overview

Frontend (Streamlit)
⇅ REST API
Backend (FastAPI + PyTorch model)


- **Frontend (`frontend_app/`)**  
  Uploads images, handles patching, calls the backend, and displays output.

- **Backend (`model_api/`)**  
  Hosts the deep learning model and performs patch-wise inference.

---

## 🧰 Tech Stack

| Layer     | Tools                           |
|-----------|----------------------------------|
| Model     | PyTorch, Timm (Swin Transformer) |
| Backend   | FastAPI, Uvicorn                 |
| Frontend  | Streamlit                        |
| Container | Docker                           |
| Versioning | Git                             |

---

## ⚙️ Prerequisites

Ensure the following are installed:

- Python 3.9+
- pip
- Docker + Docker Compose
- Git

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/wrong24/MSIGenerator.git
cd MSIGenerator
```

### 2. Add the Pre-trained Model
The pre-trained model best_generator.pth is not included due to file size.

You can either:

Train your own (see Training)

Or download it and place it like this:

MSIGenerator/
└── model_api/
    └── best_generator.pth
    
▶️ Running the Application
1️⃣ Start the Backend (FastAPI + Docker)
```bash
cd model_api/
```

# Build Docker image
docker build -t msi-generator-api .

# Run with CPU
docker run -d -p 8000:8000 --name msi_api msi-generator-api

# Or run with GPU (requires NVIDIA runtime)
docker run -d --gpus all -p 8000:8000 --name msi_api msi-generator-api
Visit: http://localhost:8000
You should see:

{"status": "MSI Generation API is running."}
2️⃣ Start the Frontend (Streamlit)
```bash
cd frontend_app/
```

# Optional: Setup virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
Visit: http://localhost:8501

🧠 Training a New Model
You can train a model on your own multispectral dataset.

1. Prepare Dataset
Organize your dataset as expected in dataset.py

Set the correct path in config.py via DATASET_ROOT_PATH

2. Adjust Training Config
Edit config.py to modify:

NUM_EPOCHS

BATCH_SIZE

LEARNING_RATE

Any other hyperparameters

3. Run Training
```bash
python train.py
```
The best model is saved as best_generator.pth.
Move it to the model_api/ folder to use it with the API.

📁 Project Structure
```arduino
MSIGenerator/
├── model_api/
│   ├── best_generator.pth
│   ├── main.py
│   ├── models.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend_app/
│   ├── app.py
│   └── requirements.txt
├── model_training/
│   ├── train.py
│   ├── config.py
│   └── dataset.py
└── README.md
```
