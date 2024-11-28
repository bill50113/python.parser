import os

import cv2
import keras.models

import numpy as np


from keras.applications.vgg19 import preprocess_input

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import matplotlib.pyplot as plt

model= keras.models.load_model('mask_19')

plt.rcParams['font.sans-serif']=['Microsoft JhengHei']
path='./images'
for i,file in enumerate(os.listdir(path)):# enumerate(列舉)檔案數/檔名--->等等要用子圖呈現
    if not file.endswith('.jpg'):continue
    img=cv2.imdecode(np.fromfile(os.path.join(path,file),dtype=np.uint8),cv2.IMREAD_COLOR)
    img=img[:,:,::-1].copy()
    img_224=cv2.resize(img,(224,224),interpolation=cv2.INTER_LINEAR)

    x=np.expand_dims(img_224,axis=0)# 增加一個維度存放答案
    x=preprocess_input(x)# 預處理圖片,將每個特徵減掉該特徵的平均數
    result=model.predict(x)# 預測
    if result[0][0]>result[0][1]:
        txt='未戴口罩'
    else:
        txt='有戴口罩'

    ax=plt.subplot(4,4,i+1)# 子圖呈現 4*4
    ax.set_title(txt)
    ax.imshow(img)
    ax.axis('off')# 關閉解析度顯示
plt.show()