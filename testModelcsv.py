import pandas as pd
import joblib

# Load the trained model
model_filename = input("Enter the filename of the trained model (e.g., model.joblib): ")
model = joblib.load(model_filename)

# Load the dataset containing extracted features
input_file = input("Enter the path of the input CSV file with extracted features: ")
data = pd.read_csv(input_file)

# Check the first few rows of the dataframe
print(data.head())

# Ensure the necessary features are present and drop any columns not needed for prediction
# Assuming 'status' is the target variable and the URL column is not included
X = data.drop(columns=['status'], errors='ignore')  # Drop the 'status' column if it exists
print("Features used for prediction:")
print(X.columns)

# Encode any categorical features in X, if they exist
X_encoded = pd.get_dummies(X)

# Ensure that the features used for prediction match the features the model was trained on
X_encoded = X_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

# Make predictions
predictions = model.predict(X_encoded)

# Add predictions to the dataframe
data['predicted_status'] = predictions

# Save the predictions to a new CSV file
output_file = input("Enter the output CSV file path to save predictions (e.g., predictions.csv): ")
data.to_csv(output_file, index=False)

print("Predictions saved successfully to:", output_file)
