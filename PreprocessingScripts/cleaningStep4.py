import pandas as pd
from urllib.parse import urlparse
import re
from math import log2

# Load keywords for feature extraction
keywords = [
    'account', 'login', 'secure', 'verify', 'update',
    'alert', 'confirm', 'suspend', 'password', 'action'
]

# Function to check if the URL contains suspicious keywords
def check_keywords(url, keywords):
    found_keywords = [keyword for keyword in keywords if keyword in url.lower()]
    return ', '.join(found_keywords) if found_keywords else None

# Function to calculate URL entropy
def calculate_entropy(url):
    if not url:
        return 0
    probability = [url.count(c) / len(url) for c in set(url)]
    return -sum(p * log2(p) for p in probability)

# Function to check for IP address in the URL
def contains_ip_address(url):
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    return int(bool(re.search(ip_pattern, url)))

# Function to extract features from a URL
def extract_features(url):
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname
    tld = hostname.split('.')[-1] if hostname and '.' in hostname else None

    # Initialize feature dictionary
    features = {
        'length_of_url': len(url),
        'hostname_length': len(hostname) if hostname else 0,
        'path_length': len(parsed_url.path) if parsed_url.path else 0,
        'query_length': len(parsed_url.query) if parsed_url.query else 0,
        'num_dots': url.count('.'),
        'contains_https': int(parsed_url.scheme == 'https'),
        'num_slashes': url.count('/'),
        'num_hyphens': url.count('-'),
        'contains_at_symbol': int('@' in url),
        'contains_subdomain': int(hostname.count('.') > 1) if hostname else 0,
        'unsafeKeywords': check_keywords(url, keywords),  # Keywords found in the URL
        'top_level_domain': tld,  # Extracted TLD
        'contains_ip_address': contains_ip_address(url),  # Check for IP address
        'num_query_parameters': len(parsed_url.query.split('&')) if parsed_url.query else 0,  # Number of query parameters
        'contains_numeric': int(any(char.isdigit() for char in url)),  # Check for numeric characters
        'num_special_chars': sum(1 for char in url if not char.isalnum() and char not in ['-', '_', '.']),  # Count special characters
        'url_entropy': calculate_entropy(url),  # URL entropy
        'length_of_hostname': len(hostname) if hostname else 0,  # Length of hostname
        'is_long_url': int(len(url) > 75),  # Is URL longer than 75 characters?
        'num_path_segments': len(parsed_url.path.split('/')) - 1,  # Number of path segments
    }

    return features

# Main program execution
if __name__ == "__main__":
    # Ask user for input file and output file names
    input_file = input("Enter the input CSV file path (e.g., 'urls.csv'): ")
    output_file = input("Enter the output CSV file path (e.g., 'output.csv'): ")

    # Load data from the specified CSV file
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print("The specified input file was not found. Please check the path and try again.")
        exit()

    # Check if the required 'URL' column exists
    if 'URL' not in df.columns:
        print("The input CSV must contain a 'URL' column.")
        exit()

    # Apply feature extraction to each URL in the dataset
    features_df = df['URL'].apply(lambda x: pd.Series(extract_features(x)))

    # Combine the original dataframe with the features
    df = pd.concat([df, features_df], axis=1)

    # Save the updated dataframe with features to the specified output file
    df.to_csv(output_file, index=False)

    # Display the updated dataframe with features
    print("Feature extraction complete. The results are saved to:", output_file)
    print(df)
