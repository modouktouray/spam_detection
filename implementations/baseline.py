import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def load_and_preprocess_data(file_path, compression="zip", index_col="Message ID"):
    data = pd.read_csv(file_path, compression=compression, index_col=index_col)
    data_length = len(data)
    print(f"Length of Data: {data_length}")
    data['Message'].fillna('', inplace=True)
    return data

def most_frequent_class(train_labels):
    class_counts = defaultdict(int)
    for label in train_labels:
        class_counts[label] += 1
    return max(class_counts, key=class_counts.get)

def evaluate_baseline(test_labels, most_frequent_class_label):
    test_data_length = len(test_labels)
    print(f"Length of Test Data: {test_data_length}")

    # Predict using the most frequent class
    predictions = [most_frequent_class_label] * test_data_length

    # Evaluate the model
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, pos_label=most_frequent_class_label)
    recall = recall_score(test_labels, predictions, pos_label=most_frequent_class_label)
    f1 = f1_score(test_labels, predictions, pos_label=most_frequent_class_label)


    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

def baseline_main():
    # Load and preprocess training data
    train_data = load_and_preprocess_data("data/train_data.zip")

    # Split data into train and test sets
    train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42)

    # Extract labels
    train_labels = train_data['Spam/Ham'].tolist()
    test_labels = test_data['Spam/Ham'].tolist()

    # Train the most frequent class baseline model
    most_frequent_class_label = most_frequent_class(train_labels)

    # Evaluate the baseline model on test data
    evaluate_baseline(test_labels, most_frequent_class_label)

