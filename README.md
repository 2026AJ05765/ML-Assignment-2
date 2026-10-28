# Machine Learning Assignment 2: Classification Model Comparison

## Student Details

- **Student Name:** Adarsh Tayade
- **BITS ID:** 2026AJ05765
- **Programme:** M.Tech Artificial Intelligence and Machine Learning
- **Assignment:** Machine Learning Assignment 2

## 1. Problem Statement

The objective of this assignment is to implement and compare five machine learning classification algorithms using the same dataset. The models are evaluated using Accuracy, Area Under the ROC Curve (AUC), Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

The assignment also includes saving trained models and developing a Streamlit application to evaluate the models using uploaded test data.

## 2. Dataset Description

The Breast Cancer Wisconsin Diagnostic dataset is used for this classification task. It is available through the scikit-learn library.

- **Total instances:** 569
- **Input features:** 30 numerical features
- **Target column:** 1
- **Missing values:** None
- **Problem type:** Binary classification

The dataset contains measurements computed from digitized images of breast mass samples.

The target labels are:

- `0`: Malignant
- `1`: Benign

The dataset was divided into training and testing sets using an 80:20 split. Stratified sampling was used to maintain the class distribution.

- **Training instances:** 455
- **Testing instances:** 114

StandardScaler was used for Logistic Regression, KNN, and Support Vector Machine to scale the input features.

## 3. Machine Learning Models

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

All models were trained on the training dataset and evaluated on the same test dataset.

## 4. Evaluation Metrics

The following metrics were calculated to compare model performance:

- **Accuracy:** Measures the proportion of correctly classified instances.
- **AUC:** Measures the model's ability to distinguish between the two classes using predicted probabilities.
- **Precision:** Measures the proportion of positive predictions that are correct.
- **Recall:** Measures the proportion of actual positive instances correctly identified.
- **F1 Score:** Represents the harmonic mean of Precision and Recall.
- **MCC:** Measures the overall quality of binary classifications using all four confusion matrix outcomes.

For Precision, Recall, and F1 Score, class `1` (Benign) is treated as the positive class.

## 5. Results and Model Comparison

The following table contains the results obtained from the Jupyter Notebook execution.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

## 6. Observations

### Logistic Regression

Logistic Regression achieved an accuracy of 0.9825 and an AUC of 0.9954. Its F1 Score was 0.9861, and its MCC was 0.9623. It achieved the highest AUC among the five models.

### Decision Tree

The Decision Tree achieved an accuracy of 0.9123 and an F1 Score of 0.9286. It had the lowest accuracy and MCC among the five models in this experiment.

### K-Nearest Neighbors (KNN)

KNN achieved an accuracy of 0.9561 and an F1 Score of 0.9655. It showed good classification performance, although its results were lower than those of Logistic Regression and SVM.

### Gaussian Naive Bayes

Naive Bayes achieved an accuracy of 0.9386 and an AUC of 0.9878. Its AUC was high, but its accuracy and F1 Score were lower than those of Logistic Regression and SVM.

### Support Vector Machine (SVM)

SVM achieved an accuracy of 0.9825 and an F1 Score of 0.9861. Its precision, recall, and MCC matched Logistic Regression. Its AUC was slightly lower than Logistic Regression.

### Overall Comparison

Logistic Regression and SVM achieved the highest accuracy, F1 Score, and MCC in this experiment. Logistic Regression had the highest AUC at 0.9954. Based on these test results, Logistic Regression is the preferred model, although SVM performed equally on the other reported metrics.

These findings apply to this dataset and this train-test split; they do not guarantee that the same model will perform best on other datasets.

## 7. Project Files

The project contains the following files:

- `ML_Assignment2.ipynb`: Notebook containing dataset preparation, model training, evaluation, and observations.
- `app.py`: Streamlit application for selecting a trained model and evaluating uploaded test data.
- `requirements.txt`: Python package dependencies.
- `test_data.csv`: Test dataset used to evaluate the trained models.
- `models/`: Directory containing the five saved models and the fitted scaler.

The Streamlit application loads the saved models and evaluates uploaded data. It does not retrain models when a test CSV is uploaded.

## 8. GitHub Repository

**Repository Link:** Add your GitHub repository URL here.

## 9. Streamlit Application

**Live Application Link:** Add your deployed Streamlit Community Cloud URL here.

The application allows users to select a classification model, upload a test CSV file, and view evaluation metrics, a confusion matrix, and a classification report.

The uploaded CSV must contain all 30 required input features and a `target` column for evaluation.

## 10. Installation and Execution

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

Run the Streamlit application using:

```bash
streamlit run app.py
```

Ensure that the `models` directory and its saved model files are available in the project directory.

## 11. Conclusion

Five classification algorithms were implemented and evaluated using the Breast Cancer Wisconsin Diagnostic dataset. Logistic Regression and Support Vector Machine achieved the highest accuracy of 0.9825. Logistic Regression achieved the highest AUC of 0.9954.

This experiment demonstrates how different classification algorithms can be compared using multiple evaluation metrics rather than relying on accuracy alone.
