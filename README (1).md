Hand Gesture Recognition Project



This project implements hand gesture recognition using \*MediaPipe\* and \*TensorFlow\*.  

It processes images from the \*ASL Alphabet dataset\* to classify American Sign Language hand gestures.  



👉 The project combines \*Deep Learning (CNN models)\* with \*Computer Vision (OpenCV + MediaPipe)\*  

to achieve \*real-time hand gesture recognition\* from webcam video streams.



These instructions ensure compatibility across Windows, macOS, and Linux.

1. Clone the repository:

git clone https://github.com/AhmedZaki10/Hand\_Gesture\_Project.git



2\. Navigate to the project directory:
cd Hand\_Gesture\_Project



3\. Install Python 3.11 (64-bit):

Download and install Python 3.11.9 (64-bit) from python.org.



-For Windows: 

Use the installer and select "Custom Installation". Install to a specific path (e.g., C:\\Python311) and avoid adding to system PATH to prevent conflicts with other Python versions.


-for macOS/Linux:

Use a package manager (e.g., brew install python@3.11 on macOS or sudo apt install python3.11 on Ubuntu) or download from python.org.


Verify the installation:python3.11 --version

Expected output: Python 3.11.9.



4\. Create a virtual environment:

Use Python 3.11 to create an isolated environment:python3.11 -m venv Hand\_Gesture\_Project



Note: Replace python3.11 with the full path (e.g., C:\\Python311\\python.exe on Windows) if python3.11 is not in your PATH.



5\. Activate the virtual environment:

-for Windows:

Hand\_Gesture\_Project\\Scripts\\activate



-for macOS/Linux:

source Hand\_Gesture\_Project/bin/activate



After activation, you should see (Hand\_Gesture\_Project) in your terminal prompt.



6.Install dependencies:

Install the required packages listed in requirements.txt:pip install -r requirements.txt



Note:



The requirements.txt includes:
-mediapipe==0.10.21
-tensorflow==2.16.1
-numpy==2.2.0
-Other dependencies compatible with Python 3.11.



If you encounter dependency conflicts, ensure pip is up-to-date:pip install --upgrade pip





7.Download the dataset:

Download the ASL Alphabet dataset from Kaggle.
Extract the asl\_alphabet\_train and asl\_alphabet\_test folders into the Hand\_Gesture\_Project directory (i.e., D:\\DEPI\_Project\\asl\_alphabet\_train and D:\\DEPI\_Project\\asl\_alphabet\_test on Windows).



8.Run the project:

Execute the main script:python main.py





Troubleshooting

Python version issues:

Ensure you’re using Python 3.11 (64-bit). MediaPipe does not support Python 3.12+ or 32-bit Python.
If multiple Python versions are installed, specify the full path to Python 3.11 when creating the virtual environment (e.g., C:\\Python311\\python.exe -m venv Hand\_Gesture\_Project).



Dependency conflicts:

If pip install -r requirements.txt fails, try installing dependencies individually:pip install mediapipe==0.10.21 tensorflow==2.16.1 numpy==2.2.0



If conflicts persist, clear the virtual environment and reinstall:deactivate
rmdir /s Hand\_Gesture\_Project  # Windows
rm -rf Hand\_Gesture\_Project   # macOS/Linux
python3.11 -m venv Hand\_Gesture\_Project
source Hand\_Gesture\_Project/bin/activate  # or Hand\_Gesture\_Project\\Scripts\\activate
pip install -r requirements.txt





MediaPipe import errors:

Verify MediaPipe installation:python -c "import mediapipe as mp; print(mp.**version**)"

Expected output: 0.10.21.
If you see ImportError related to protobuf or tensorflow, ensure tensorflow==2.16.1 and protobuf==4.25.8 are installed.



Dataset issues:

Ensure the asl\_alphabet\_train and asl\_alphabet\_test folders are directly under the project directory and contain the correct image structure.



Project Structure

main.py: Main script for running the hand gesture recognition model.
requirements.txt: Lists all required Python packages.
README.md: Project setup instructions.
.gitignore: Ignores virtual environment and temporary files.
asl\_alphabet\_train/: Training dataset folder (to be downloaded).
asl\_alphabet\_test/: Test dataset folder (to be downloaded).

Notes

The project uses mediapipe==0.10.21 for hand gesture detection and tensorflow==2.16.1 for model processing, both compatible with Python 3.11.
Ensure your system has at least 8GB RAM and sufficient disk space for the dataset and virtual environment.
For macOS Apple Silicon (M1/M2/M3), use pip install mediapipe-silicon instead of mediapipe if you encounter compatibility issues.

