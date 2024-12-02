# Phishing URL Detection: Data and Model Repository

This repository contains resources, scripts, and models for detecting phishing URLs based on data collected between 2019 and 2024. The goal is to build a robust system for identifying malicious URLs using feature extraction and machine learning models.

## Repository Structure

- **Phishing URLs Files (2019-2024)**: Contains datasets of known phishing URLs collected over the years.
- **Safe URLs File (300k)**: A large dataset of safe URLs to balance the phishing URL dataset.

### Data Cleaning Scripts
1. **`cleaningStep1.py` and `cleaningStep2.py`**:
   - Remove unnecessary columns from datasets.
   - Label phishing URLs as malicious and safe URLs as non-malicious.

2. **`cleaningStep3.py`**:
   - Merge the phishing URL dataset with an equal number of safe URLs.
   - Transform datasets from monthly to annual collections.

3. **`cleaningStep4.py`**:
   - Extract features from URLs:
     - **Length**: URL, hostname, path, query.
     - **Counts**: Dots, slashes, hyphens, query parameters, special characters.
     - **Flags**: Presence of HTTPS, `@`, subdomains, IP addresses, numeric characters.
     - **Keywords**: Presence of suspicious words.
     - **TLD**: Top-level domain.
     - **Entropy**: Measure of randomness in the URL.
     - **Long URL**: Flag for URLs over 75 characters.

### Preprocessed Data
- **Preprocessed Data (2019-2024)**: Cleaned and feature-extracted datasets ready for training and testing.

---

## Model Training and Testing

### Training Scripts
1. **`trainModel.py`**:
   - Train a Logistic Regression model using the preprocessed 2019 dataset.
   - Perform 1000 iterations to optimize the model.
   - Save the trained model as `model2019.joblib`.

2. **`trainExistingModel.py`**:
   - Iteratively train the 2019 model on datasets from 2020-2023.
   - Save the updated model as `model2020-2023.joblib`.

### Testing Scripts
1. **`testModelcsv.py`**:
   - Test the model using the 2024 dataset.
   - Evaluate F1 score, accuracy, recall, and precision metrics.

2. **`csvToF1Metric.py`**:
   - Measure F1 metrics for iterative model training steps.

---

## Iterative Model Training Workflow
1. Start with the 2019 dataset for initial training.
2. Use new data from subsequent years (2020-2023) to refine the model iteratively.
3. Test and evaluate the final model on the 2024 dataset.

---

## Feature Extraction and Preprocessing Steps
The feature extraction process ensures that URLs are represented in a way that enhances model learning. Key features include:
- **Structural Characteristics**: Length, counts of various characters.
- **Security Indicators**: HTTPS, presence of subdomains, IP addresses.
- **Semantic Indicators**: Suspicious keywords, entropy, and TLD.

---

## Dependencies
Ensure the following Python libraries are installed:
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage Instructions

1. Clone the repository:
 ```bash
 git clone <repository-url>
 cd <repository-folder>
```

2. Run the cleaning scripts (cleaningStep1.py to cleaningStep4.py) to preprocess the datasets.

3. Train the initial model:
 ```bash
python trainModel.py
```

4. Iteratively train on new datasets:
 ```bash
python trainExistingModel.py
```

5.Test the model:
```bash
python testModelcsv.py
```
## Contributions

Feel free to fork the repository, make changes, and submit pull requests if you have any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

   
