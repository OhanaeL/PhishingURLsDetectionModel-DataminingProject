import pandas as pd

# Ask for the input and output file names
input_file = input("Enter the name of the input CSV file (with .csv extension): ")
output_file = input("Enter the name of the output CSV file (with .csv extension): ")

# Load the dataset
df = pd.read_csv(input_file)

# Rename the existing column to 'URL'
df.columns = ['URL']

# Add the new 'status' column with all values set to 'safe'
df['status'] = 'safe'

# Save the updated dataframe to the specified output CSV file
df.to_csv(output_file, index=False)

print("Column renamed to 'URL' and 'status' column added successfully!")
