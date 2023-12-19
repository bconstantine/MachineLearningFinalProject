import os as os
import numpy as np
save_dir = './data'
import tensorflow.keras as keras
import csv
def get_data(datapath):
    pos_files = os.listdir(datapath + '/pos')
    neg_files = os.listdir(datapath + '/neg')
    #print(len(pos_files))
    #print(len(neg_files))

    pos_all = []
    neg_all = []
    for pf, nf in zip(pos_files, neg_files):
        with open(datapath + '/pos' + '/' + pf, encoding='utf-8') as f:
            s = f.read()
            pos_all.append(s)
        with open(datapath + '/neg' + '/' + nf, encoding='utf-8') as f:
            s = f.read()
            neg_all.append(s)

    X_orig= np.array(pos_all + neg_all)
    Y_orig = np.array([1 for _ in range(len(pos_all))] + [0 for _ in range(len(neg_all))])
    #print("X_orig:", X_orig.shape)
    #print("Y_orig:", Y_orig.shape)

    return X_orig, Y_orig

def get_data_(datapath):
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
    X_orig= np.array(pos_all + neg_all)
    Y_orig = np.array([1 for _ in range(len(pos_all))] + [0 for _ in range(len(neg_all))])

    return X_orig, Y_orig

vocab_size = 30000

def generate_train_vector():
    X_orig, Y_orig = get_data('/home/cjweibb/workspace/MachineLearning/FinalProj/aclImdb/train')
    X_orig_test, Y_orig_test = get_data('/home/cjweibb/workspace/MachineLearning/FinalProj/aclImdb/test')

    X_orig = np.concatenate([X_orig,get_data_("")[0], X_orig_test])
    Y_orig = np.concatenate([Y_orig, get_data_("")[1],Y_orig_test])
    print(len(X_orig), len(Y_orig))
    maxlen = 200

    t = keras.preprocessing.text.Tokenizer(vocab_size)
    t.fit_on_texts(X_orig)  
    word_index = t.word_index  
    v_X = t.texts_to_sequences(X_orig)  

    pad_X = keras.preprocessing.sequence.pad_sequences(v_X, maxlen=maxlen, padding='post')

    np.savez(save_dir+'/all_train_vector_Data', x=pad_X, y=Y_orig)
    import copy
    x = list(t.word_counts.items())
    s = sorted(x, key=lambda p: p[1], reverse=True)
    small_word_index = copy.deepcopy(word_index) 
    for item in s[vocab_size:]:
        small_word_index.pop(item[0])
    np.save(save_dir+'/small_word_index', small_word_index)


if __name__ == '__main__':
    generate_train_vector()
