import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Data Loading
df = pd.read_csv("Telco-Customer-Churn.csv")

# Data Cleaning

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df["TotalCharges"] = df["TotalCharges"].fillna(0)

df = df.drop("customerID", axis=1)

X = df.drop("Churn", axis=1)

y = df["Churn"]
print("X shape:", X.shape)
print("y shape:", y.shape)
y = y.map({"No": 0, "Yes": 1})
print(y.value_counts())

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size = 0.2,
    random_state = 42,
    stratify = y
)
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Preprocessing

numeric_features = X.select_dtypes(exclude="object").columns
categorical_features = X.select_dtypes(include="object").columns

print("Numerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Models

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ]
)

decision_tree_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(random_state=42))
    ]
)

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ]
)

knn_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]
)

svm_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", SVC())
    ]
)


gradient_boosting_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", GradientBoostingClassifier(random_state=42))
    ]
)

models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "KNN": knn_model,
    "SVM": svm_model,
    "Gradient Boosting": gradient_boosting_model
}


results = []


for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })


results_df = pd.DataFrame(results)

print("\nModel Performance:")
print(results_df.to_string(index=False))


#5-FOLD CROSS-VALIDATION


cv_results = []


for name, model in models.items():

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="f1"
    )

    cv_results.append({
        "Model": name,
        "Mean CV F1": cv_scores.mean()
    })


cv_results_df = pd.DataFrame(cv_results)

print("\nCross-Validation Results:")
print(cv_results_df.to_string(index=False))


#Hypertuning

param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}


grid_search = GridSearchCV(
    logistic_model,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)


grid_search.fit(X_train, y_train)


print("\nBest Parameters:")
print(grid_search.best_params_)

print("Best CV F1-score:")
print(grid_search.best_score_)

#Final Model


final_model = grid_search.best_estimator_

y_pred_final = final_model.predict(X_test)


accuracy_final = accuracy_score(
    y_test,
    y_pred_final
)

precision_final = precision_score(
    y_test,
    y_pred_final
)

recall_final = recall_score(
    y_test,
    y_pred_final
)

f1_final = f1_score(
    y_test,
    y_pred_final
)


print("\nFinal Model Performance:")
print("Accuracy:", accuracy_final)
print("Precision:", precision_final)
print("Recall:", recall_final)
print("F1-score:", f1_final)


#Confusion Matrix
cm_final = confusion_matrix(
    y_test,
    y_pred_final
)

print("\nFinal Confusion Matrix:")
print(cm_final)



# CHURN PROBABILITY
y_probability = final_model.predict_proba(X_test)

print("\nFirst 10 Churn Probabilities:")
print(y_probability[:10, 1])

#  NEW CUSTOMER PREDICTION
new_customer = pd.DataFrame({
    "gender": ["Female"],
    "SeniorCitizen": [0],
    "Partner": ["Yes"],
    "Dependents": ["No"],
    "tenure": [5],
    "PhoneService": ["Yes"],
    "MultipleLines": ["No"],
    "InternetService": ["Fiber optic"],
    "OnlineSecurity": ["No"],
    "OnlineBackup": ["Yes"],
    "DeviceProtection": ["No"],
    "TechSupport": ["No"],
    "StreamingTV": ["Yes"],
    "StreamingMovies": ["Yes"],
    "Contract": ["Month-to-month"],
    "PaperlessBilling": ["Yes"],
    "PaymentMethod": ["Electronic check"],
    "MonthlyCharges": [85.50],
    "TotalCharges": [427.50]
})


new_prediction = final_model.predict(
    new_customer
)

new_probability = final_model.predict_proba(
    new_customer
)[0][1]


print("\nNew Customer Prediction:")
print(
    "Predicted Churn:",
    "Yes" if new_prediction[0] == 1 else "No"
)

print(
    f"Churn Probability: {new_probability * 100:.2f}%"
)






