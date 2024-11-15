import pandas as pd
import os

# Ask the user to input the directory where CSV files are located
folder_path = input("Please enter the path to the folder containing the CSV files: ")

# Verify if the provided directory exists
if not os.path.exists(folder_path):
    print("The provided folder path does not exist.")
else:
    # List all CSV files in the specified folder
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if not file_list:
        print("No CSV files found in the specified folder.")
    else:
        # Create an empty DataFrame to store the merged data
        merged_data = pd.DataFrame()

        # Loop through the files and append them to the merged_data DataFrame
        for file in file_list:
            file_path = os.path.join(folder_path, file)
            # Read the current CSV file into a DataFrame
            df = pd.read_csv(file_path)
            # Append the data to the merged DataFrame
            merged_data = pd.concat([merged_data, df], ignore_index=True)

        # Get the directory name (last part of the input path) to use in the result file name
        directory_name = os.path.basename(os.path.normpath(folder_path))
        
        # Create the output file name based on the input directory
        output_file = f'./merged_{directory_name}.csv'
        
        # Save the merged data as a CSV file
        merged_data.to_csv(output_file, index=False)

        print(f'Merged file saved as {output_file}')
