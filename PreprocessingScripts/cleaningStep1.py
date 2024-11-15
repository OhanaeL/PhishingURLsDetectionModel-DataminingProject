import pandas as pd

# Ask for the input and output file names
input_file = input("Enter the name of the input CSV file (with .csv extension): ")
output_file = input("Enter the name of the output CSV file (with .csv extension): ")

# Load the dataset
df = pd.read_csv(input_file)

# Remove the two columns (replace 'date' and 'description' with actual column names if different)
df = df.drop(['date', 'description'], axis=1)

# Keep only the 'URL' column
df = df[['URL']]

# Add the new 'status' column with all values set to 'malicious'
df['status'] = 'malicious'

# Save the updated dataframe to the specified output CSV file
df.to_csv(output_file, index=False)

print("Columns removed and 'status' column added successfully!")
