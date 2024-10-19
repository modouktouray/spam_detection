# Enron Dataset Processing

This dowloaad_enron_dataset and data_processing scripts downloads and processes the Enron Spam Data, separating it into training and testing datasets. It also saves the datasets to zip files for further use.

These scripts should be run independently.

### How to run the different implementations
-python3 main.py {model_type}

# Model types:
    -naive_bayes
    -logistic_regression
    -rnn
    -svm
# Example: To run naive_bayes:
    python3 main.py naive_bayes