import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_and_preprocess_data(file_path, compression="zip", index_col="Message ID"):
    data = pd.read_csv(file_path, compression=compression, index_col=index_col)
    data_length = len(data)
    print(f"Length of Data: {data_length}")
    data['Message'].fillna('', inplace=True)
    return data

def preprocess_text_data(data, max_words, max_len):
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(data['Message'])

    sequences = tokenizer.texts_to_sequences(data['Message'])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data['Spam/Ham'])

    return padded_sequences, labels, tokenizer, label_encoder

def build_rnn_model(max_words, max_len):
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=32, input_length=max_len))
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_rnn_model(data, max_words=1000, max_len=100, test_size=0.2, epochs=5):
    X, y, tokenizer, label_encoder = preprocess_text_data(data, max_words, max_len)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model = build_rnn_model(max_words, max_len)

    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

    return model, tokenizer, label_encoder

def evaluate_rnn_model(model, tokenizer, label_encoder, test_data):
    test_sequences = tokenizer.texts_to_sequences(test_data['Message'])
    padded_test_sequences = pad_sequences(test_sequences, maxlen=model.input_shape[1], padding='post', truncating='post')

    predictions = model.predict(padded_test_sequences)
    predictions = (predictions > 0.5).astype(int)

    # Convert predictions back to original labels
    predictions_labels = label_encoder.inverse_transform(predictions.flatten())

    # Evaluation metrics
    accuracy = accuracy_score(test_data['Spam/Ham'], predictions_labels)
    precision = precision_score(test_data['Spam/Ham'], predictions_labels, pos_label='spam')
    recall = recall_score(test_data['Spam/Ham'], predictions_labels, pos_label='spam')
    f1 = f1_score(test_data['Spam/Ham'], predictions_labels, pos_label='spam')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

def rnn_main():
    # Load and preprocess training data
    train_data = load_and_preprocess_data("data/train_data.zip")

    # Train the RNN model
    trained_rnn_model, trained_tokenizer, label_encoder = train_rnn_model(train_data)

    # Load and preprocess test data
    test_data = load_and_preprocess_data("data/test_data.zip")

    # Evaluate the RNN model on test data
    evaluate_rnn_model(trained_rnn_model, trained_tokenizer, label_encoder, test_data)

