This is a comprehensive and structured `README.md` for your **Customer Churn Prediction** project, based on the provided source files.

---

# Customer Churn Prediction using Artificial Neural Networks (ANN)

This is an end-to-end machine learning project designed to predict whether a bank customer is likely to churn (leave the bank) based on various demographic and financial factors. The project utilizes a deep learning approach with an Artificial Neural Network (ANN) and provides a user-friendly interface via a Streamlit web application.

## 📊 Project Overview

The goal is to classify customers into two categories: those likely to churn and those likely to stay. The model is trained on the `Churn_Modelling.csv` dataset, which includes features like credit score, geography, gender, age, tenure, and account balance.

## 🛠️ Key Features

* **Data Preprocessing**: Includes dropping irrelevant features (`RowNumber`, `CustomerId`, `Surname`), encoding categorical variables, and feature scaling.
* **Deep Learning Model**: A multi-layer Sequential ANN built with TensorFlow/Keras.
* **Interactive Web App**: A Streamlit-based dashboard where users can input customer data to get real-time predictions.
* **Training Monitoring**: Integrated with TensorBoard for visual tracking of accuracy and loss during training.

## 🏗️ Model Architecture

The ANN is constructed with the following layers:

* **Input Layer**: Accepts 12 features (after one-hot encoding for Geography).
* **Hidden Layer 1**: 64 neurons with ReLU activation.
* **Hidden Layer 2**: 32 neurons with ReLU activation.
* **Output Layer**: 1 neuron with Sigmoid activation for binary classification.
* **Optimization**: Adam optimizer with a learning rate of 0.1 and Binary Crossentropy loss function.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed. You can install the required dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt

```

### Running the Application

To launch the Streamlit web interface, run the following command in your terminal:

```bash
streamlit run app.py

```

## 📁 Project Structure

* `app.py`: The Streamlit application script containing the UI and prediction logic.
* `cleaning.ipynb`: Jupyter notebook detailing data cleaning, encoding, model building, and training.
* `Churn_Modelling.csv`: The dataset used for training and testing.
* `model.h5`: The trained Keras model.
* `artifacts/`: Contains serialized preprocessing objects:
* `label_encoder_gender.pkl`: Encoder for the 'Gender' feature.
* `onehot_encoder_geo.pkl`: Encoder for the 'Geography' feature.
* `scaler.pkl`: StandardScaler for feature normalization.


* `logs/`: Directory for TensorBoard training logs.

## 🧪 Technologies Used

* **Frameworks**: TensorFlow, Keras, Streamlit
* **Data Handling**: Pandas, NumPy
* **Machine Learning**: Scikit-learn (for scaling and encoding)
* **Visualization**: TensorBoard, Matplotlib

---