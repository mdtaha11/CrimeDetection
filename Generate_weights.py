import numpy as np
import os

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3,preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

#Transfer learning using pretrained Inception V3
def inception(img):
    
    inception_base=InceptionV3(weights='imagenet',include_top=True,input_shape=(299,299,3))
    
    m=Model(inputs=inception_base.input,outputs=inception_base.get_layer('avg_pool').output)

    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    x=preprocess_input(x)
    
    return m.predict(x)

#weights for our crime video frames
#put path variable as the location of your Crime video frames
crime_weights=[]
images=[]
for j in range(1,41):
    path=r'D:\DATASET\Crime video frames\{}'.format(j) 
    for i in os.listdir(path):
        print(os.path.join(path,i))
        img=(image.load_img(os.path.join(path,i),target_size=(299,299)))
        crime_weights.append(inception(img)) 

#weights for our normal video frames
#put path variable as the location of your Normal video frames
normal_weights=[]
images=[]
for j in range(1,31):
    path=r'D:\DATASET\Normal video frames\{}'.format(j) 
    for i in os.listdir(path):
        print(os.path.join(path,i))
        img=(image.load_img(os.path.join(path,i),target_size=(299,299)))
        normal_weights.append(inception(img)) 






















