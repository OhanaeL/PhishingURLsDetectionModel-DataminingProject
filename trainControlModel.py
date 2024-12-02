import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Initialize an empty DataFrame to store the merged data
data = pd.DataFrame()

# Allow the user to input multiple CSV files
print("Enter the paths of the CSV files to be merged. Enter a blank line when done.")

while True:
    input_file = input("Enter the path of a CSV file (or press Enter to finish): ")
    if input_file == "":
        break
    try:
        # Load the current CSV file
        temp_data = pd.read_csv(input_file)
        # Append it to the main DataFrame
        data = pd.concat([data, temp_data], ignore_index=True)
        print(f"Loaded {len(temp_data)} rows from {input_file}.")
    except Exception as e:
        print(f"Error loading file {input_file}: {e}")

# Ensure data is not empty
if data.empty:
    print("No data loaded. Exiting.")
    exit()

# Check the first few rows of the merged dataset
print(f"Merged dataset contains {len(data)} rows.")
print(data.head())

# Assuming 'status' is the target variable and the rest are features
X = data.drop(columns=['status', 'URL'])  # Drop the URL column and the target variable
y = data['status']  # Target variable

# Encode any categorical features in X, if they exist
X_encoded = pd.get_dummies(X)

# Create the logistic regression model
model = LogisticRegression(max_iter=1000)

# Train the model on the full dataset
print("Training the model...")
model.fit(X_encoded, y)

# Generate predictions for the full dataset
y_pred = model.predict(X_encoded)

# Calculate accuracy
accuracy = accuracy_score(y, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Print the classification report
print("Classification Report:")
print(classification_report(y, y_pred))

# Compute confusion matrix
conf_matrix = confusion_matrix(y, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot Confusion Matrix as Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix Heatmap")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Save the trained model
model_filename = input("Enter the filename to save the model (e.g., model.joblib): ")
joblib.dump(model, model_filename)

print("Model saved successfully.")
