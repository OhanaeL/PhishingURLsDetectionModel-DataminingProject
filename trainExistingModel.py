import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the saved model
model_filename = input("Enter the filename of the saved model (e.g., model.joblib): ")
model = joblib.load(model_filename)

print("Model loaded successfully.")

# Load the new training data
new_data_file = input("Enter the path of the new data CSV file: ")
new_data = pd.read_csv(new_data_file)

# Display the first few rows of the new data to confirm
print("First few rows of the new data:")
print(new_data.head())

# Separate features and target variable
X_new = new_data.drop(columns=['status', 'URL'])  # Drop 'URL' and target 'status'
y_new = new_data['status']

# Encode categorical features if needed
X_new_encoded = pd.get_dummies(X_new)

# Align columns with the loaded model's expected input
X_new_encoded = X_new_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

# Train the loaded model on the new data
print("Training the model with new data...")
model.fit(X_new_encoded, y_new)

# Evaluate the updated model (optional)
X_train, X_test, y_train, y_test = train_test_split(X_new_encoded, y_new, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)

# Print evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Updated Model Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the updated model
updated_model_filename = input("Enter the filename to save the updated model (e.g., updated_model.joblib): ")
joblib.dump(model, updated_model_filename)
print(f"Updated model saved successfully as {updated_model_filename}.")
