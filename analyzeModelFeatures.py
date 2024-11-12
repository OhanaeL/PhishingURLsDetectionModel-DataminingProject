import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression  # Import LogisticRegression

# Load the saved model
model_filename = input("Enter the filename of the saved model (e.g., model.joblib): ")
model = joblib.load(model_filename)

# Ensure the model is logistic regression
if not isinstance(model, LogisticRegression):
    raise TypeError("The loaded model is not a Logistic Regression model.")

# Retrieve feature names and coefficients
feature_names = model.feature_names_in_
coefficients = model.coef_[0]  # Access the coefficients for the single logistic regression model

# Combine feature names with their coefficients
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

# Calculate absolute values of coefficients to rank their impact
feature_importance['Importance'] = np.abs(feature_importance['Coefficient'])

# Sort features by importance (absolute coefficient values)
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Display the feature importance
print("Feature Importance (sorted by absolute coefficient values):")
print(feature_importance[['Feature', 'Coefficient', 'Importance']])

# Highlight the most significant features
print("\nTop features contributing to predictions:")
print(feature_importance.head(50))  # Change the number to see more features if needed
