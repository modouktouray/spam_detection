import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def load_and_preprocess_data(file_path, compression="zip", index_col="Message ID"):
    data = pd.read_csv(file_path, compression=compression, index_col=index_col)
    data_length = len(data)
    print(f"Length of Data: {data_length}")
    data['Message'].fillna('', inplace=True)
    
    print("Total Count")
    print(data["Spam/Ham"].value_counts(dropna=False))
    print("\nProportion in %")
    print(round(data["Spam/Ham"].value_counts(normalize=True), 4)*100)
    return data

def train_model(data, vectorizer_max_features=1000, cv_folds=5, top_features=50, k_best=1000):
    vectorizer = CountVectorizer(max_features=vectorizer_max_features, stop_words='english')
    X_train_counts = vectorizer.fit_transform(data['Message'])

    # Feature selection using chi-squared test
    selector = SelectKBest(chi2, k=k_best)
    X_train_selected = selector.fit_transform(X_train_counts, data['Spam/Ham'])

    clf = MultinomialNB()
    clf.fit(X_train_selected, data['Spam/Ham'])

    # Cross-validation
    cv_scores = cross_val_score(clf, X_train_selected, data['Spam/Ham'], cv=cv_folds)
    print("Cross-Validation Scores:", cv_scores)
    print("Mean CV Accuracy:", cv_scores.mean())

    # Get the selected feature indices
    selected_feature_indices = selector.get_support(indices=True)

    feature_names = vectorizer.get_feature_names_out()
    
    print("\nTop", top_features, "Selected Features:")
    for idx in selected_feature_indices[:top_features]:
        print(feature_names[idx])

    return clf, vectorizer

def evaluate_model(model, vectorizer, test_data):
    test_data_length = len(test_data)
    print(f"Length of Test Data: {test_data_length}")
    test_data['Message'].fillna('', inplace=True)

    # Transform the test data using the same vectorizer
    X_test_counts = vectorizer.transform(test_data['Message'])

    # Predict on the test set
    predictions = model.predict(X_test_counts)

    # Evaluate the model
    accuracy = accuracy_score(test_data['Spam/Ham'], predictions)
    precision = precision_score(test_data['Spam/Ham'], predictions, pos_label='spam')
    recall = recall_score(test_data['Spam/Ham'], predictions, pos_label='spam')
    f1 = f1_score(test_data['Spam/Ham'], predictions, pos_label='spam')

    # Print confusion matrix
    conf_matrix = confusion_matrix(test_data['Spam/Ham'], predictions)
    print("Confusion Matrix:\n", conf_matrix)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

   
def naive_bayes_main():
     # Load and preprocess training data
    train_data = load_and_preprocess_data("data/train_data.zip")

    # Train the model
    trained_model, trained_vectorizer = train_model(train_data)

    # Load and preprocess test data
    test_data = load_and_preprocess_data("data/test_data.zip")

    # Evaluate the model on test data
    evaluate_model(trained_model, trained_vectorizer, test_data)
