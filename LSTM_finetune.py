from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers
import pandas as pd
import string
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import f1_score, confusion_matrix
import json
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import numpy as np

nltk.download('wordnet')

####### sentiment_labelled_sentences
train_df = pd.read_csv('../sentiment_labelled_sentences/combined_train.csv')
test_df = pd.read_csv('../sentiment_labelled_sentences/combined_test.csv')
val_df = pd.read_csv('../sentiment_labelled_sentences/combined_val.csv')


######### ACLImdb dataset ################3
imdb_train = pd.read_csv('../aclImdb/train.csv')
imdb_train = imdb_train.drop(['decimalLabel'], axis=1)

stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

train_df['processed_text'] = train_df.text.apply(lambda x: clean_text(x))
val_df['processed_text'] = val_df.text.apply(lambda x: clean_text(x))

imdb_train['processed_text'] = imdb_train.text.apply(lambda x: clean_text(x))

# temp = train_df.processed_text.apply(lambda x: len(x.split(" "))).mean()
# print(temp)

max_features = 6000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train_df['processed_text'])
list_tokenized_train = tokenizer.texts_to_sequences(train_df['processed_text'])
list_tokenized_val = tokenizer.texts_to_sequences(val_df['processed_text'])

list_tokenized_imdb_train = tokenizer.texts_to_sequences(imdb_train['processed_text'])

maxlen = 130
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
y = train_df['integerLabel']
X_val = pad_sequences(list_tokenized_val, maxlen=maxlen)
y_val = val_df['integerLabel']

imdb_X = pad_sequences(list_tokenized_imdb_train, maxlen=maxlen)
imdb_y = imdb_train['integerLabel']

embed_size = 128
batch_size = 128
epochs = 10
n_splits = 10
skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def create_model():
    model = Sequential()
    model.add(Embedding(max_features, embed_size))
    model.add(Bidirectional(LSTM(32, return_sequences = True)))
    model.add(GlobalMaxPool1D())
    model.add(Dense(20, activation="relu"))
    model.add(Dropout(0.03))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_model()
# model.fit(imdb_X, imdb_y, batch_size=batch_size, epochs=2, validation_split=0.2)
model.fit(X_t,y, batch_size=64, epochs=5, validation_data=(X_val, y_val))

test_df["text"]=test_df.text.apply(lambda x: clean_text(x))
y_test = test_df["integerLabel"]
list_sentences_test = test_df["text"]
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

train_loss, train_acc = model.evaluate(X_t, y, verbose=0)
print(f'Training Accuracy: {train_acc}')

# Evaluate on validation set
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f'Validation Accuracy: {val_acc}')

# Evaluate on test set
test_loss, test_acc = model.evaluate(X_te, y_test, verbose=0)
print(f'Test Accuracy: {test_acc}')

scores_per_fold5 = []
for train_index, val_index in skf5.split(X_t, y):
    X_train_fold, X_val_fold = X_t[train_index], X_t[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]

    model = create_model()
    model.fit(X_train_fold, y_train_fold, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(X_val_fold, y_val_fold))
    
    # Evaluate the model
    _, accuracy = model.evaluate(X_te, y_test, verbose=0)
    scores_per_fold5.append(accuracy)

# Print cross-validation results
mean_accuracy = np.mean(scores_per_fold5)
std_accuracy = np.std(scores_per_fold5)
print(f'5 fold Cross-Validation Accuracy: {mean_accuracy:.4f} +/- {std_accuracy:.4f}')


scores_per_fold10 = []
for train_index, val_index in skf10.split(X_t, y):
    X_train_fold, X_val_fold = X_t[train_index], X_t[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]

    # model = create_model()
    # model.fit(X_train_fold, y_train_fold, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(X_val_fold, y_val_fold))
    
    # Evaluate the model
    _, accuracy = model.evaluate(X_te, y_test, verbose=0)
    scores_per_fold10.append(accuracy)

# Print cross-validation results
mean_accuracy = np.mean(scores_per_fold10)
std_accuracy = np.std(scores_per_fold10)
print(f'10 fold Cross-Validation Accuracy: {mean_accuracy:.4f} +/- {std_accuracy:.4f}')