# pip install scikit-learn nltk gensim xlrd openpyxl pandas tensorflow==2.10.1 matplotlib
# pip install scipy==1.10.1 (這裡不一定要裝,有問題再裝)
# 情緒監控模型上遇到的難題:
# 1.資料集非常龐大
# 2.非常耗費人力進行標示
# 依照要監控的目標類型去標記分類
# 1:罵習維尼
# 2:罵金正恩
# 3:罵....
# 99:罵....
import os
import pickle
import re
import shutil
import time

import nltk
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras import Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.integration_test.preprocessing_test_utils import BATCH_SIZE
from keras.layers import Embedding, Dropout, LSTM, Dense
from keras.utils import pad_sequences
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def preprocess(txt):
    txt = re.sub(text_cleaning_re, ' ', str(txt).lower()).strip()
    # re(不規則表示式) str(txt)先轉成小寫-->去除text_cleaning_re所設定的字元字串-->轉成''空白-->strip()去除\n換行
    tokens = []
    for token in txt.split():
        if token not in stop_words:
            tokens.append(token)
    return " ".join(tokens)
columns = ["target", "ids", "date", "flag", "user", "text"]
text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
nltk.download('stopwords')
stop_words = stopwords.words("english")
df=pd.read_csv(
    'training.1600000.processed.noemoticon.csv',
    encoding='ISO-8859-1',
    names=columns
)
print('去除停用詞....')

df.text=df.text.apply(lambda x:preprocess(x))
with open("eng_dic.pkl", 'rb') as file: # 載入做好的字典
    tok=pickle.load(file)
df_train, df_test, =train_test_split(df, test_size=0.2, random_state=1)
# 會再回傳 y值 所以只有設定 df_train,df_test 後面 ,留空格就好
vocab_size=len(tok.word_index)

weights=np.zeros([vocab_size,100])
# 取得w2v(英文向量化) 模型中每個字的100個權重(可調整)
w2v_model=Word2Vec.load('w2v_model')
for word, i in tok.word_index.items():
    # w2v 會刪除停用詞以及出現次數小於10次的錯別字,但tok字典並沒有去除' 出現次數小於10次的錯別字 '
    if word in w2v_model.wv:# 所以這邊在檢查 tok字典裡的字有沒有出現在 w2v模型裡面,有的話就
        weights[i]=w2v_model.wv[word]# 把權重放進去 weights , 沒有的話權重就會都是 0,因為上面weights=np.zeros
# 處理要訓練的句子
x_train=pad_sequences(tok.texts_to_sequences(df_train.text), maxlen=300)
# pad_sequences :　擴充長度,讓每個字串同長度
x_test=pad_sequences(tok.texts_to_sequences(df_test.text), maxlen=300)
# pad_sequences :　擴充長度,讓每個字串同長度
# 處理結果 : 0 負面的 , 4正面的

encoder = LabelEncoder()
encoder.fit(df_train.target.tolist())# target就是'training.1600000.processed.noemoticon.csv'
#                                       文件裡面 0 跟 4 的標記,做成list送進去encoder做訓練

y_train=encoder.transform(df_train.target.tolist()) # 製作訓練的結果
y_test=encoder.transform(df_test.target.tolist()) # 製作測試的結果
y_train=y_train.reshape(-1, 1)#轉成 n*1 維度
y_test=y_test.reshape(-1, 1)#轉成 n*1 維度


model=Sequential()
#                      每一層都是一個演算法,所以沒有固定SOP,依照自己所學去調整

model.add(
    Embedding(  # 嵌入式模型 https://ithelp.ithome.com.tw/articles/10328879
        vocab_size,
        100,#每個字100個權重
        weights=[weights], # 將前面設定的 weight放進去
        input_length=300,
        trainable=False
    )
)
model.add(Dropout(0.5))# 防止過擬合
model.add(LSTM(100,dropout=0.2, recurrent_dropout=0.2))# LSTM長短期記憶
model.add(Dense(1, activation='sigmoid'))# 扁平層
model.compile(
    loss='binary_crossentropy', # 二元交叉火商
    optimizer='adam', # 優化器
    metrics=['accuracy'] # 以精準度為主
)


# callback 是在模型訓練時,要回調的函數(在這邊是兩個函數的集合(list)),還沒丟給模型
# ReduceLROnPlateau : Reduce Learning Rate: 逐步將學習率降低,後面講優化器時會詳細說明
# EarlyStopping : 訓練時,若損失值不再減少(已經逼近極值),就提早停止(預防浪費後面不必要的訓練時間)
# Yolo7/8 才終於加入 EarlyStopping
callbacks=[
    ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
    EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)
]
print('開始訓練模型....')


BATCH_SIZE=1024 # OOM時,請調小(記憶體不足)
t1=time.time()
history=model.fit( # 保留歷史資料--->訓練結果圖表化,看精準度有沒有增加,損失值有沒有減少,有沒有收斂
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=8,
    validation_split=0.1, # 10%用於測試資料使用
    verbose=1,#輸出進度條
    validation_data=(x_test, y_test),
    callbacks=callbacks# 執行上面callbacks[裡面的兩支函數]
)
t2=time.time()
print(f'訓練時間：{t2-t1}秒')
if os.path.exists('sentiment_model'):
    shutil.rmtree('sentiment_model')
model.save('sentiment_model')

score=model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print(f"Loss : {score[0]}")
print(f"Accuracy : {score[1]}")
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']
x=range(len(acc))
plt.plot(x, acc, 'b', label='Training Acc')
plt.plot(x, val_acc, 'g', label='Validation Acc')
plt.plot(x, loss, 'r', label='Train loss')
plt.plot(x, val_loss, 'y', label='Validation loss')
plt.legend()#顯示圖例
plt.show()





# 老師訓練好的模型 https://mahaljsp.ddns.net/files/nlp/sentiment_model.zip


















