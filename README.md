# Customer Churn Prediction System

## Project Overview

This project builds a machine learning system to predict whether a telecom customer is likely to churn.

The project uses the IBM Telco Customer Churn dataset containing 7,043 customer records. The workflow includes data cleaning, preprocessing, model training, evaluation, cross-validation, hyperparameter tuning, and prediction for new customers.

## Dataset

The dataset contains customer information such as:

* Gender
* Senior Citizen status
* Tenure
* Internet Service
* Contract type
* Payment Method
* Monthly Charges
* Total Charges
* Churn

The target variable is **Churn**:

* `0` → No Churn
* `1` → Churn

## Technologies Used

* Python
* Pandas
* Scikit-learn

## Project Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Feature & Target Separation
   ↓
Train-Test Split
   ↓
Data Preprocessing
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Cross-Validation
   ↓
Hyperparameter Tuning
   ↓
Final Model
   ↓
New Customer Prediction
```

## Data Preprocessing

The following preprocessing steps were performed:

* Converted `TotalCharges` to numeric values.
* Handled missing values in `TotalCharges`.
* Removed `customerID` because it is only an identifier.
* Converted the `Churn` target into binary values.
* Applied `StandardScaler` to numerical features.
* Applied `OneHotEncoder` to categorical features.

A Scikit-learn `Pipeline` and `ColumnTransformer` were used to keep preprocessing and model training together.

## Machine Learning Models

The following classification models were trained and evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Gradient Boosting

## Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

5-fold cross-validation was also performed using F1-score.

### Model Performance

| Model               | Accuracy | Precision | Recall | F1-score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |   80.55% |    65.72% | 55.88% |   60.40% |
| Decision Tree       |   72.11% |    47.55% | 49.20% |   48.36% |
| Random Forest       |   78.64% |    62.54% | 48.66% |   54.74% |
| KNN                 |   76.37% |    55.27% | 57.49% |   56.36% |
| SVM                 |   79.13% |    64.18% | 48.40% |   55.18% |
| Gradient Boosting   |   80.27% |    66.55% | 51.60% |   58.13% |

## Hyperparameter Tuning

GridSearchCV was used to tune the Logistic Regression model.

The `C` parameter was tested using:

```text
0.01, 0.1, 1, 10, 100
```

The best parameter was:

```text
C = 10
```

The best cross-validation F1-score was approximately:

```text
59.89%
```

## Final Model Performance

The tuned Logistic Regression model achieved:

* **Accuracy:** 80.48%
* **Precision:** 65.52%
* **Recall:** 55.88%
* **F1-score:** 60.32%

### Confusion Matrix

```text
[[925 110]
 [165 209]]
```

## New Customer Prediction

The final model was also tested with a new customer record.

Example output:

```text
Predicted Churn: Yes
Churn Probability: 83.73%
```

The churn probability represents the model's estimated probability for the given customer.

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Open the project folder

```bash
cd Customer_Churn_Prediction
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python main.py
```

## Project Structure

```text
Customer_Churn_Prediction/
│
├── Telco-Customer-Churn.csv
├── main.py
├── requirements.txt
└── README.md
```
