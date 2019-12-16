from TextClassification import TextClassification
import pandas as pd
import DataPreprocess as DP
from numpy import *
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

sess = tf.InteractiveSession()

data = pd.read_csv(r'e:/train_poi.csv', encoding='utf8')
x = data['type3']
y = [[str(i)] for i in data['二级土地_1']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

preprocess = DP.DataPreprocess()

data_type = 'multiple'
'''''
clf = TextClassification()
texts_seq, texts_labels = clf.get_preprocess(x_train, y_train, 
                                             word_len=1, 
                                             num_words=500,
                                             sentence_len=6)
clf.fit(texts_seq=texts_seq,
        texts_labels=texts_labels,
        output_type=data_type,
        epochs=100,
        batch_size=64,
        model=None)

# 保存整个模块,包括预处理和神经网络
with open(r'e:/./%s.pkl' % data_type, 'wb') as f:
    pickle.dump(clf, f)
'''''
# 导入刚才保存的模型
with open(r'e:/./%s.pkl' % data_type, 'rb') as f:
    clf = pickle.load(f)
y_predict = clf.predict(x_test)

y_predict = [[clf.preprocess.label_set[i.argmax()]] for i in y_predict]
for i in y_predict:
    print(i)
print(len(y_predict))
print(y_test)
print(y_predict)
score = sum(y_predict == np.array(y_test)) / len(y_test)
print(score)  # 0.9288





































