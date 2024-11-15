import pandas as pd

# Load the entire safe URLs dataset
safe_df = pd.read_csv('cleaned_safe.csv')

# Keep track of how many safe URLs have been used
used_safe_count = 0

# List of malicious files to process (add more as needed)
malicious_files = [
    'cleaned_phishing_2019.csv',
    'cleaned_phishing_2020.csv',
    'cleaned_phishing_2021.csv',
    'cleaned_phishing_2022.csv',
    'cleaned_phishing_2023.csv',
    'cleaned_phishing_2024.csv'
]

# Process each malicious file
for malicious_file in malicious_files:
    # Load the malicious data
    malicious_df = pd.read_csv(malicious_file)

    # Determine the number of malicious entries
    num_malicious = len(malicious_df)

    # Select an equal number of unique safe URLs that haven't been used yet
    safe_subset = safe_df.iloc[used_safe_count:used_safe_count + num_malicious]

    # Update the count of used safe URLs
    used_safe_count += num_malicious

    # Combine the malicious and safe data
    combined_df = pd.concat([malicious_df, safe_subset])

    # Randomize the order of the combined dataset
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)

    # Save the combined dataset to a new CSV file
    output_file = f'mixed_dataset_{malicious_file}'
    combined_df.to_csv(output_file, index=False)

    print(f"Created {output_file} with {num_malicious} malicious and {num_malicious} safe URLs, randomized.")

# Check if all safe URLs have been used
if used_safe_count >= len(safe_df):
    print("Warning: All safe URLs have been used.")
else:
    print(f"{len(safe_df) - used_safe_count} safe URLs remain unused.")
