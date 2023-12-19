import os as os
import numpy as np
save_dir = './data'
import tensorflow.keras as keras
import csv

def get_data():
    pos_all = []
    neg_all = []
    
    with open('/home/cjweibb/workspace/MachineLearning/FinalProj/sentiment labelled sentences/combined_test.csv', 'r') as file:
        csv_reader = csv.reader(file)
        
        for row in csv_reader:
            if row[1] == '1':
                pos_all.append(row[0])
                
            else:
                neg_all.append(row[0])
    with open('/home/cjweibb/workspace/MachineLearning/FinalProj/sentiment labelled sentences/combined_val.csv', 'r') as file:
        csv_reader = csv.reader(file)
    
        for row in csv_reader:
            if row[1] == '1':
                pos_all.append(row[0])
            else:
                neg_all.append(row[0])
    with open('/home/cjweibb/workspace/MachineLearning/FinalProj/sentiment labelled sentences/combined_train.csv', 'r') as file:
        csv_reader = csv.reader(file)
    
        for row in csv_reader:
            if row[1] == '1':
                pos_all.append(row[0])
            else:
                neg_all.append(row[0])    
    print(pos_all)
    X_orig= np.array(pos_all + neg_all)
    Y_orig = np.array([1 for _ in range(len(pos_all))] + [0 for _ in range(len(neg_all))])

    return X_orig, Y_orig

vocab_size = 30000

def generate_train_vector():
    X_orig, Y_orig = get_data()

    maxlen = 200
    t = keras.preprocessing.text.Tokenizer(vocab_size) 
    t.fit_on_texts(X_orig) 
    word_index = t.word_index 
    v_X = t.texts_to_sequences(X_orig) 
    pad_X = keras.preprocessing.sequence.pad_sequences(v_X, maxlen=maxlen, padding='post')


    np.savez(save_dir+'/sent_train_vector_Data', x=pad_X, y=Y_orig)
    import copy
    x = list(t.word_counts.items())
    s = sorted(x, key=lambda p: p[1], reverse=True)
    small_word_index = copy.deepcopy(word_index)

    for item in s[vocab_size:]:
        small_word_index.pop(item[0])

    print(len(small_word_index))
    print(len(word_index))
    np.save(save_dir+'/small_word_index', small_word_index)


if __name__ == '__main__':
    generate_train_vector()
