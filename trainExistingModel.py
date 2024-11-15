import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

# Ask for the number of entries to use for training
num_entries = input("Enter the number of entries to use for training (leave blank to use all entries): ")

if num_entries:
    num_entries = int(num_entries)
    train_data = new_data.head(num_entries)  # Use only the specified number of entries for training
    test_data = new_data.tail(len(new_data) - num_entries)  # Use the remaining data for testing
else:
    train_data = new_data  # If no number is specified, use all data for training
    test_data = pd.DataFrame()  # No separate test data

# Separate features and target variable for training
X_train = train_data.drop(columns=['status', 'URL'])  # Drop 'URL' and target 'status'
y_train = train_data['status']

# Separate features and target variable for testing (if applicable)
X_test = test_data.drop(columns=['status', 'URL'])  # Drop 'URL' and target 'status'
y_test = test_data['status']

# Encode categorical features if needed
X_train_encoded = pd.get_dummies(X_train)
X_test_encoded = pd.get_dummies(X_test)

# Align columns with the loaded model's expected input
X_train_encoded = X_train_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
X_test_encoded = X_test_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

# Train the loaded model on the new data
print("Training the model with new data...")
model.fit(X_train_encoded, y_train)

# If there's test data, evaluate the updated model
if not test_data.empty:
    y_pred = model.predict(X_test_encoded)

    # Print evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Updated Model Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Visualization: Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# Save the updated model
updated_model_filename = input("Enter the filename to save the updated model (e.g., updated_model.joblib): ")
joblib.dump(model, updated_model_filename)
print(f"Updated model saved successfully as {updated_model_filename}.")
