from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Activation



#creating datasets of weights by concatenating crime_weights and normal_weight
weights=crime_weights+normal_weights
arr_weights=np.array([i[0].reshape(2048) for i in weights])
y=[1 for i in range(len(crime_weights))]+[0 for i in range(len(normal_weights))] 

#We need the weights in the array of form (Total number of videos i.e 40+30, 40,2048)
data=arr_weights.reshape(len(weights)//40,40,2048)     #len(weights)//40 = 70


#Converting y into categorical values
from tensorflow.keras.utils import to_categorical
y_binary = to_categorical(y)

#Splitiing our dataset
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(data,y_binary,test_size=0.2)

#RNN
chunk_size=2048
n_chunk=40
rnn_size=512
model=Sequential()
model.add(LSTM(512,input_shape=(n_chunk,chunk_size)))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('sigmoid'))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
epoch=1500
batch_size=100


model.fit(X_train,y_train, epochs=epoch,batch_size=100 )     #This step might take time depending 
#on your system's performance
 
test_result=model.predict(X_test)>0.5
from sklearn.metrics import accuracy_score    
accuracy_score(y_test,test_result)   #We achieved an accurace of 92.8%

#You can try with different epoch and batch_sizes or different model architecture to achieve more accuracy 


