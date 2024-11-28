import os
import pickle

import keras
from keras.utils import pad_sequences

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def getLabel(score):
    label='中立的'
    if score <= 0.4:
        label='負面的'
    elif score >= 0.7:
        label='正向的'
    return label

with open('eng_dic.pkl','rb') as file:
    tok=pickle.load(file)
model=keras.models.load_model('sentiment_model')
while True: # 輸入後判斷句子 正向/負向
    txt=input('請輸入英文句子 : ')
    if txt=='quit':break
    x_test=pad_sequences(tok.texts_to_sequences([txt]),maxlen=300)# pad_sequences :　擴充長度,讓每個字串同長度
    score=model.predict([x_test])[0] # 二維的([0]對應到分數)
    label=getLabel(score)
    print(score,label)















































