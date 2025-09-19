# Hand Gesture Recognition Project  

This project implements **hand gesture recognition** using [MediaPipe](https://developers.google.com/mediapipe) and [TensorFlow](https://www.tensorflow.org/). It processes images from the **ASL Alphabet dataset** to classify American Sign Language hand gestures.  

The setup instructions below ensure compatibility across **Windows, macOS, and Linux**.  

---

## 🚀 Features  
- Real-time **hand gesture detection** using MediaPipe  
- **Deep learning classification** with TensorFlow  
- Compatible with the **ASL Alphabet dataset**  
- Fully reproducible with Python **3.11 (64-bit)**  

---

## 📦 Setup Instructions  

### 1. Clone the Repository  
```bash
git clone https://github.com/AhmedZaki10/Hand_Gesture_Project.git
cd Hand_Gesture_Project
```

---

### 2. Install Python 3.11 (64-bit)  
- **Windows**:  
  - Download and install [Python 3.11.9 (64-bit)](https://www.python.org/downloads/release/python-3119/).  
  - Choose *Custom Installation* → install to a specific path (e.g., `C:\Python311`).  
  - ⚠️ Do **not** add to system PATH to avoid conflicts.  

- **macOS/Linux**:  
  ```bash
  # macOS (Homebrew)
  brew install python@3.11  

  # Ubuntu/Debian
  sudo apt install python3.11
  ```

- Verify installation:  
  ```bash
  python3.11 --version
  ```
  Expected output:  
  ```
  Python 3.11.9
  ```

---

### 3. Create a Virtual Environment  
```bash
python3.11 -m venv Hand_Gesture_Project
```
> On Windows, you may need to specify the full path, e.g.:
> ```bash
> C:\Python311\python.exe -m venv Hand_Gesture_Project
> ```

---

### 4. Activate the Virtual Environment  
- **Windows**:  
  ```bash
  Hand_Gesture_Project\Scripts\activate
  ```  
- **macOS/Linux**:  
  ```bash
  source Hand_Gesture_Project/bin/activate
  ```  

Once activated, your terminal should show:  
```bash
(Hand_Gesture_Project) $
```

---

### 5. Install Dependencies  
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Key dependencies (predefined in `requirements.txt`):  
- `mediapipe==0.10.21`  
- `tensorflow==2.16.1`  
- `numpy==2.2.0`  

---

### 6. Download Dataset  
Download the **ASL Alphabet dataset** from [Kaggle](https://www.kaggle.com/grassknoted/asl-alphabet).  

Extract the dataset into the project directory:  
```
Hand_Gesture_Project/
│── asl_alphabet_train/
│── asl_alphabet_test/
```

---

### 7. Run the Project  
```bash
python main.py
```

---

## 🛠 Troubleshooting  

### Python version issues  
- Ensure you are using **Python 3.11 (64-bit)**.  
- MediaPipe is **not compatible** with Python 3.12+ or 32-bit installations.  

---

### Dependency conflicts  
If installation fails:  
```bash
pip install mediapipe==0.10.21 tensorflow==2.16.1 numpy==2.2.0
```
If issues persist:  
```bash
# Reset environment
deactivate
rm -rf Hand_Gesture_Project   # macOS/Linux
rmdir /s Hand_Gesture_Project # Windows

# Recreate environment
python3.11 -m venv Hand_Gesture_Project
source Hand_Gesture_Project/bin/activate  # or Scripts\activate on Windows
pip install -r requirements.txt
```

---

### MediaPipe import errors  
Check installation:  
```bash
python -c "import mediapipe as mp; print(mp.__version__)"
```
Expected output: `0.10.21`  

If `protobuf` errors occur:  
```bash
pip install protobuf==4.25.8
```

For **Apple Silicon (M1/M2/M3)**:  
```bash
pip install mediapipe-silicon
```

---

### Dataset issues  
Ensure:  
- `asl_alphabet_train/` and `asl_alphabet_test/` are in the **project root directory**.  
- The folder structure matches the dataset format (each class in its own subfolder).  

---

## 📂 Project Structure  

```
Hand_Gesture_Project/
│── main.py                # Main script
│── requirements.txt       # Dependencies
│── README.md              # Documentation
│── .gitignore             # Ignore virtual env & temp files
│── asl_alphabet_train/    # Training dataset (downloaded)
│── asl_alphabet_test/     # Testing dataset (downloaded)
```

---

## 💡 Notes  
- Uses **MediaPipe (0.10.21)** for detection & **TensorFlow (2.16.1)** for model training.  
- Requires at least **8GB RAM** and sufficient disk space.  
- Compatible across **Windows, macOS, Linux**.  

---

✅ You are now ready to run **Hand Gesture Recognition with ASL Alphabet dataset**!  
