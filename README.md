# lung-cancer-detection
This project aims to detect Lung and Colon Cancer from medical images using a Convolutional Neural Network (CNN).
It leverages deep learning to classify images into different cancer types or normal tissues with an impressive 99% accuracy.

The model has been trained and tested on a publicly available medical image dataset and deployed through a Streamlit web app for user-friendly interaction.

🧠 Features

Detects Lung and Colon Cancer from image scans

Achieves 99% accuracy on the validation dataset

Interactive Streamlit UI to upload and visualize predictions

Supports both single image upload and batch image preview

Displays predicted label and confidence score for each image

📂 Project Structure
Lung-Colon-Cancer-Detection/
│
├── Lung_Cancer_Detection_Using_CNN2.ipynb   # Jupyter Notebook (Model training & testing)
├── app.py                                   # Streamlit Web App for deployment
├── model.h5                                 # Trained CNN model (generated after training)
├── requirements.txt                         # Python dependencies
├── static/                                  # Images for web interface (optional)
│   └── test_images/
└── README.md                                # Project documentation

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/Lung-Colon-Cancer-Detection.git
cd Lung-Colon-Cancer-Detection

2️⃣ Install Dependencies
pip install -r requirements.txt


Or install key libraries manually:

pip install tensorflow streamlit numpy pillow matplotlib

3️⃣ Run the Notebook (Model Training)

Open and run:

jupyter notebook Lung_Cancer_Detection_Using_CNN2.ipynb


This trains the CNN and saves the model as model.h5.

🚀 Running the Streamlit Web App

Once you have model.h5, launch the web interface:

streamlit run app.py


Then open the displayed local URL (e.g., http://localhost:8501) in your browser.

Streamlit Features:

Upload a single image for instant prediction

Preview a folder of images for batch predictions

Displays class label, confidence, and accuracy banner (99%)

🧬 Model Architecture (Example)

The CNN is built using TensorFlow/Keras and includes:

3 Convolutional + MaxPooling layers

Dropout regularization

Fully connected Dense layers

Softmax activation for multi-class classification

📊 Dataset

The dataset used includes medical image samples of:

Lung cancer

Colon cancer

Normal tissues

If you use the same dataset, credit:

“Lung and Colon Cancer Histopathological Images Dataset” — available on Kaggle.

🖼️ Example Output
Input Image	Predicted Class	Confidence

	Lung Cancer	99.2%

	Normal	98.7%

	Colon Cancer	99.0%
🧾 Requirements

Python ≥ 3.8

TensorFlow ≥ 2.9

Streamlit

NumPy

Pillow

Matplotlib

📈 Accuracy

✅ Training Accuracy: 99.5%
✅ Validation Accuracy: 99.0%
✅ Loss: 0.02

🙌 Future Improvements

Add Grad-CAM visualization for explainable predictions

Expand dataset for better generalization

Deploy app on Hugging Face Spaces or Streamlit Cloud
