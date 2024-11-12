import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import joblib

# Load the dataset
input_file = input("Enter the path of the input CSV file: ")
data = pd.read_csv(input_file)

# Check the first few rows of the dataframe
print(data.head())

# Assuming 'status' is the target variable and the rest are features
X = data.drop(columns=['status', 'URL'])  # Drop the URL column and the target variable
y = data['status']  # Target variable

# Encode any categorical features in X, if they exist
X_encoded = pd.get_dummies(X)

# Create the logistic regression model
model = LogisticRegression(max_iter=2000)

# Perform 10-fold cross-validation and get accuracy scores for each fold
cv_scores = cross_val_score(model, X_encoded, y, cv=10, scoring='accuracy')

# Calculate mean accuracy across folds
mean_accuracy = np.mean(cv_scores)
print(f"10-Fold Cross-Validation Mean Accuracy: {mean_accuracy:.2f}")

# Plot accuracy scores across folds
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), cv_scores, marker='o', linestyle='-', color='b', label='Accuracy per Fold')
plt.axhline(y=mean_accuracy, color='r', linestyle='--', label=f'Mean Accuracy: {mean_accuracy:.2f}')
plt.xticks(range(1, 11))
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("10-Fold Cross-Validation Accuracy per Fold")
plt.legend()
plt.grid(True)
plt.show()

# Train the model on the full dataset and save it
model.fit(X_encoded, y)
model_filename = input("Enter the filename to save the model (e.g., model.joblib): ")
joblib.dump(model, model_filename)

print("Model saved successfully.")


