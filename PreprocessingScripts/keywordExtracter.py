import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Step 1: Load Your CSV File
file_path = "merged_2024.csv"  # Replace with your file path
url_column = "URL"  # Replace with the actual column name that contains URLs

# Optional: Use chunksize for large datasets
chunk_size = 10000  # Adjust the chunk size as needed based on performance
chunks = pd.read_csv(file_path, chunksize=chunk_size)

# Step 2: Tokenize URLs in Chunks
def tokenize_url(url):
    # Normalize the URL to lower case and remove common URL components
    url = url.lower()
    url = re.sub(r'https?://', '', url)  # Remove http:// or https://
    url = re.sub(r'\.com|\.net|\.org|\.edu|\.gov|\.co|\.io|\.biz|\.info|\.me|\.jp|\.cn', '', url)  # Remove common domain endings
    url = re.sub(r'www\.', '', url)  # Remove www.
    # Split the URL into tokens using common delimiters
    tokens = re.split(r"[/.?&=_\-:]", url)
    tokens = [token for token in tokens if token]  # Remove empty tokens
    return tokens

# Collect all tokens from all chunks
all_tokens = []
for chunk in chunks:
    urls = chunk[url_column].dropna().tolist()
    for url in urls:
        all_tokens.extend(tokenize_url(url))

# Step 3: Count Frequencies of All Words
word_count = Counter(all_tokens)

# Print the 10 most common words
print("Most Common Words:")
for word, count in word_count.most_common(10):  # Change 10 to see more/less
    print(f"{word}: {count}")

# Step 4: Visualize the Results
# Bar Chart of Most Common Words
plt.figure(figsize=(10, 5))
common_words = word_count.most_common(10)  # Get the 10 most common words
plt.bar([word[0] for word in common_words], [word[1] for word in common_words], color='skyblue')
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Most Common Words in Phishing URLs")
plt.show()

# Word Cloud of All Words
wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_count)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Step 5: Save the Results to a CSV File (Optional)
# Convert the word count to a DataFrame
word_df = pd.DataFrame(word_count.items(), columns=["Word", "Frequency"])

# Save the DataFrame to a new CSV file
output_path = "word_frequency_results_2024.csv"
word_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")
