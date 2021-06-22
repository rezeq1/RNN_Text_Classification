import numpy as np
import pandas as pd
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Dropout,SpatialDropout1D,GlobalMaxPooling1D,LSTM,GRU,Bidirectional
from keras.models import Sequential
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import string


#Text Preprocessing function
tokenizer1 = RegexpTokenizer('[a-zA-Z]\w+\'?\w*')
nltk.download('stopwords')

def clean_str(doc):
    # split into tokens by white space
    tokens = tokenizer1.tokenize(doc)
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out short tokens
    tokens = [word.lower() for word in tokens if len(word) > 1]
    # delete stopword
    tokens = [word for word in tokens if word not in stopwords.words()]

    return " ".join(tokens)

df = pd.read_csv("training_ex4_dl2021b.csv")

sentences=[]
for i in df['sentence'].values:
    sentences.append(clean_str(i))
    
df['sentence']=sentences

tokenizer = Tokenizer(num_words=50000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['sentence'].values)

X = tokenizer.texts_to_sequences(df['sentence'].values)
X = pad_sequences(X, maxlen=250)
Y = df['label'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1000)

model = Sequential()
model.add(Embedding(50000, 100, input_length=X.shape[1]))
model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2,return_sequences=True)))
model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2,return_sequences=False)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=3, batch_size=100,validation_data=(X_test,Y_test))

test_df=pd.read_csv("test_ex4_dl2021b.csv")

sentences_p=[]
for i in test_df['sentence'].values:
    sentences_p.append(clean_str(i))

X_predict = tokenizer.texts_to_sequences(sentences_p)
X_predict = pad_sequences(X_predict, maxlen=250)

preds=model.predict(X_predict)

# Rounding the output of the model to 0 or 1
ans=[round(i[0]) for i in preds.tolist()]
Test_pred = pd.DataFrame({"id":test_df['id'],"label":ans})
Test_pred.to_csv('Results.csv')



