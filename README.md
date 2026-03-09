# Heart-Attack-Risk-Prediction-Hybrid-XGBoost-ANN
Heart Attack Risk Prediction using a Hybrid Machine Learning Model combining XGBoost and Artificial Neural Networks (ANN) on the Cleveland Heart Disease Dataset.
#  Heart Attack Risk Prediction using Hybrid Model (XGBoost + ANN)

##  Project Overview

This project predicts the risk of a heart attack using a **Hybrid Machine Learning Model** that combines **XGBoost** and **Artificial Neural Networks (ANN)**.

The idea behind the hybrid model is to leverage the strengths of both algorithms:

* **XGBoost** captures complex patterns in structured data.
* **ANN** learns deeper feature interactions.

By combining both models, the system achieves **better predictive performance** compared to individual models.



##  Dataset

Dataset used: **Cleveland Heart Disease Dataset**

* Total Records: **303**
* Total Features: **14**

### Important Features

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate Achieved
* Exercise Induced Angina
* ST Depression (Oldpeak)
* Number of Major Vessels
* Thalassemia

### Target Variable

| Value | Meaning          |
| ----- | ---------------- |
| 0     | No Heart Disease |
| 1     | Heart Disease    |



##  Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* TensorFlow / Keras
* Matplotlib
* Seaborn



##  Hybrid Model Architecture

The hybrid architecture works as follows:

1. Input clinical features
2. Train **XGBoost model**
3. Extract **XGBoost probability output**
4. Concatenate probability with original features
5. Feed the combined features into **ANN**
6. ANN produces the final prediction

### ANN Architecture

Input Layer
↓
Dense Layer (32 neurons, ReLU)
↓
Dropout (0.2)
↓
Dense Layer (16 neurons, ReLU)
↓
Output Layer (Sigmoid)



##  Model Evaluation

The model is evaluated using multiple performance metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* Precision-Recall Curve

### Performance

Accuracy: **~90%**

ROC-AUC Score: **0.91**

This indicates strong discriminative capability for predicting heart disease risk.



##  Visualizations

The project includes several visualizations:

* XGBoost Feature Importance
* ROC Curve
* Confusion Matrix
* Precision-Recall Curve
* ANN Training vs Validation Accuracy



##  Project Workflow

1. Data Preprocessing
2. Feature Engineering
3. Feature Scaling
4. Train-Test Split
5. XGBoost Model Training
6. Hybrid ANN Model Training
7. Model Evaluation
8. Prediction for New Patient
9. Model Saving for Deployment



##  Saved Models

The trained models are saved for future inference:

* `scaler.pkl`
* `xgb_model.pkl`
* `ann_model.h5`

These files allow predictions without retraining the model.



##  Future Improvements

Possible improvements for this project:

* Explainable AI using SHAP
* Web application using Streamlit
* Deployment using Docker
* Integration with real clinical datasets
* Real-time health prediction systems



##  Author

**Diya Panwar**
Data Science / AI Enthusiast
