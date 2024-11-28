import os

import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg19 import preprocess_input

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = keras.models.load_model('flower')
path='./images'
# daisy 雛菊 , dandelion 蒲公英 , roses 玫瑰 , sunflowers 向日葵 , tulips 鬱金香
kind={0:'daisy',1:'dandelion',2:'roses',3:'sunflowers',4:'tulips'}

for i,file in enumerate(os.listdir(path)):
    img=cv2.imdecode(np.fromfile(os.path.join(path,file),dtype=np.uint8),cv2.IMREAD_COLOR)[:,:,::-1].copy()
    x=cv2.resize(img,(224,224),interpolation=cv2.INTER_LINEAR)
    x=np.expand_dims(x,axis=0)
    x=preprocess_input(x)
    out=model.predict(x)
    #print(file,kind[np.argmax(out)])
    name=kind[np.argmax(out)]
    ax=plt.subplot(5,5,i+1)
    ax.set_title(name)
    ax.imshow(img)
    ax.axis('off')
plt.show()



































