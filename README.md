# Real vs Fake Face Detection CNN

A deep learning project built with PyTorch that uses a custom Convolutional Neural Network (CNN) to classify human faces as either **Real** or **AI-Generated (Fake)**. 

The model is deployed with a user-friendly web interface using **Gradio** and is hosted on **Hugging Face Spaces**.

## 🚀 Live Demo
You can test the model directly in your browser without installing anything!
* **https://huggingface.co/spaces/Hriddo/LookSee_AI** 
## 📊 Dataset
The model was trained on the **140k Real and Fake Faces** dataset from Kaggle. 
* **Training Set:** 100,000 images (50,000 Real, 50,000 Fake)
* **Testing Set:** 20,000 images
* **Image Resolution:** Resized to 64x64 RGB pixels during preprocessing.

## 🧠 Model Architecture
The custom Deep CNN architecture consists of four distinct convolutional blocks, designed to extract hierarchical facial features from basic edges to complex AI artifacts:

* **Block 1-4:** Each block contains two `Conv2d` layers (kernel size 3x3), followed by a `ReLU` activation, `BatchNorm2d` for training stability, and `MaxPool2d` for spatial downsampling.
* **Classifier Head:** A fully connected `Linear` layer (512 neurons), accompanied by `Dropout(0.5)` to prevent overfitting, feeding into the final output layer (2 classes).
* **Optimizer:** Adam
* **Loss Function:** Cross-Entropy Loss

## 📈 Performance
The model achieved high reliability in distinguishing genuine human faces from synthetic ones.
* **Testing Accuracy:** **96.42%**

## 💻 Running the App Locally

If you want to run the Gradio web interface on your own machine:

**1. Clone the repository:**
```bash
git clone [https://github.com/hri-ddo/real-vs-fake-face-detection.git](https://github.com/hri-ddo/real-vs-fake-face-detection.git)
cd real-vs-fake-face-detection
```

**2. Install dependencies:**
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

**3. Download the Model Weights:**
*(Note: Due to GitHub file size limits, the `model_data.pkl` file is not stored in this repository. Download it from the Hugging Face space linked above and place it in the root directory).*

**4. Run the application:**
```bash
python app.py
```
A local server will start, and you can open the interface in your web browser at `http://127.0.0.1:7860`.

## 📁 File Structure
* `app.py`: The main script containing the model architecture, secure CPU unpickling logic, and the Gradio web interface.
* `requirements.txt`: List of Python dependencies required to run the project.
* `README.md`: Project documentation.
