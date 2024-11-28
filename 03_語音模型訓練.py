#pip install librosa tensorflow==2.10.1
# librosa 是處理音頻的萬能庫 EX:spa/pps/硬體壓縮(硬編)
# 下載https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge/data?select=train.7z
# 解壓縮後將 audio檔案剪出來到根目錄, 最後將 audio 裡的 _background_noise_ (背景噪音) 刪除
# preprocess.zip 公用程式
# 解壓縮後, 將 preprocess.py 至於專案根目錄底下, 獎 preprocess.py裡面程式碼的DATA_PATH路徑改成 './audio/'
import keras.losses
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.utils import to_categorical

from preprocess import get_labels, save_data_to_array, get_train_test

max_pad_len=1200 # 聲音的長度,原本建議是 11,但有測試過 這次要訓練語音 1200剛剛好
labels=get_labels()[0]
# print(labels)
nc=len(labels)# number of class , 可以辨識幾個單字,此資料集為 30 個單字
# save_data_to_array(path='./audio',max_pad_len=max_pad_len)# 將硬碟所有的聲音檔都讀進記憶體裡面,                <轉 .npy>
#                                                           再儲存成.npy檔(numpy的陣列),方便下次讀取訓練
#                                                           下次執行時這行就可以註解掉了(已儲存過npy),網路上有的人提供訓練模型時沒有提供轉npy的代碼,要自己補上

# 切割訓練組測試組
x_train,x_test,y_train,y_test=get_train_test()

# 最後拓展一個維度(axis=-1),與前面 VGG19不同是拓展第一個(axis=0)
x_train=np.expand_dims(x_train,axis=-1)
x_test=np.expand_dims(x_test,axis=-1)

# 轉為 onehot
y_train_hot=to_categorical(y_train)
y_test_hot=to_categorical(y_test)
print(y_train_hot)

# 建立模型
model=Sequential()#                               'relu'線性整流
model.add(Conv2D(32,kernel_size=(2,2),activation='relu',input_shape=(20,max_pad_len,1)))# input_shape=(20,max_pad_len,1)
model.add(MaxPooling2D(pool_size=(2,2)))                                              #        依照preprocess沒有頻譜
model.add(Dropout(0.25))# 25%捨棄,防止過擬合
model.add(Flatten())

#全連接層
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(nc,activation='softmax'))# nc(30),softmax(轉成機率)
# 宣告模型損失函數跟優化器
model.compile(
    loss=keras.losses.categorical_crossentropy,# categorical_crossentropy() (分類火商)
    optimizer=keras.optimizers.Adadelta(),# 優化器
    metrics=['accuracy']# metrics(指標),以['accuracy']準確性為主

)
model.fit(
    x_train,
    y_train_hot,
    batch_size=300,
    epochs=1000,
    verbose=1,
    validation_data=(x_test,y_test_hot)
)

score=model.evaluate(x_test,y_test_hot)
print(score)
model.save('asr.h5')























