import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

# Load the predictions from the CSV file
input_file = input("Enter the path of the predictions CSV file (e.g., predictions.csv): ")
data = pd.read_csv(input_file)

# Check the first few rows of the dataframe
print(data.head())

# Assuming the original target variable is 'status' and predictions are in 'predicted_status'
if 'status' not in data.columns or 'predicted_status' not in data.columns:
    print("The CSV must contain 'status' and 'predicted_status' columns.")
    exit()

# Extract the true labels and predicted labels
y_true = data['status']
y_pred = data['predicted_status']

# Calculate the F1 score
f1 = f1_score(y_true, y_pred, average='weighted')
accuracy = accuracy_score(y_true, y_pred)

# Print the results
print(f"F1 Score: {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_true, y_pred))

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix using Seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Malicious', 'Safe'], yticklabels=['Malicious', 'Safe'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
