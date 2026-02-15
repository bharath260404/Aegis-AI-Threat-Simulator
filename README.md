# 🛡️ Aegis: AI-Powered Threat Simulation & Detection System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/AI-PyTorch%20%26%20XGBoost-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![Status](https://img.shields.io/badge/Status-MVP%20Complete-success)

**Aegis** is a full-stack "Purple Team" cybersecurity platform that uses Artificial Intelligence to simulate and detect cyber attacks. It features a **GAN (Generative Adversarial Network)** that acts as an automated attacker ("Red Team") and an **XGBoost Classifier** that acts as the defender ("Blue Team").

  Key Features

 🔵 Blue Team (Defense)
* **Real-time Traffic Analysis:** Analyzes network packets to classify them as "Normal" or "Malicious."
* **High Accuracy:** Trained on the **NSL-KDD dataset**, achieving **~79% accuracy** on unknown test data.
* **Visual Dashboard:** Live metrics showing threat levels and blocked packets.

 🔴 Red Team (Offense)
* **AI Attack Generator:** A custom **GAN (Generator)** trained to create synthetic, zero-day attack vectors.
* **Adversarial Simulation:** Launches 100+ fake attacks instantly to stress-test the defense system.
* **Bypass Testing:** Automatically calculates how many AI-generated attacks manage to sneak past the firewall.

---

 🛠️ Installation & Setup

Since the datasets and models are large, this project includes automated scripts to regenerate them on your machine.

1. Clone the Repository
```bash
git clone [https://github.com/bharath260404/Aegis-AI-Threat-Simulator.git](https://github.com/bharath260404/Aegis-AI-Threat-Simulator.git)

cd Aegis-AI-Threat-Simulator

2. Set Up Virtual Environment
Bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
pip install torch torchvision torchaudio xgboost streamlit plotly pandas scikit-learn
⚡ How to Run
Step 1: Initialize Data & Models
First, download the data and train the AI models. (This only needs to be done once).

Bash
# 1. Download and Process Data
python src/data_loader.py
python src/preprocessing.py

# 2. Train the Defender (XGBoost)
python src/train_model.py

# 3. Train the Attacker (GAN)
python src/train_gan.py
Step 2: Launch the Dashboard
Start the web interface to interact with the AI.

Bash
streamlit run src/app.py

🧠 System Architecture
The project follows a standard Machine Learning Pipeline:

Ingestion: data_loader.py fetches the NSL-KDD dataset.

Preprocessing: preprocessing.py applies One-Hot Encoding and Normalization.

Training:

train_model.py builds the XGBoost Classifier (The Shield).

train_gan.py builds the PyTorch Generator (The Sword).

Deployment: app.py serves the models via a Streamlit frontend.
