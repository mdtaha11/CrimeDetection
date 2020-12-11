# CrimeDetection
The project explains implementation of video classification. It uses various Deep Learning tools and techniques to frame a good architecture of video classifier.
I will try to make things as much clear as I can.

The project tries to detect crime in a video by classifying videos as:-
1) Crime videos
2) Normal videos

The project is based on convolution neural networks and recurrent neural network.

Architecture of our model:-

The first network used in our project is the convolutional neural network to extract high-level features of the images. We used a pre-trained model, Inception V3 and applied transfer learning technique.
Inception V3 is developed by Google. It is trained on the ImageNet Large Visual Recognition Challenge dataset. Please refer to Google's official documentation to know more about Inception-v3.

The second network used is the recurrent neural network. In CNN, the inputs and outputs are inpendent to each other but in RNN, the outputs from previous step are fed as inputs to the current step. RNN are mailny used to 
predict the next word in a sentence, by remembering the previous words. Hidden state in RNN remembers information about the sequence.
The layer has first layer as LSTM layer followed by two hidden layers, the first one has 1024 nodes with the activation function relu and the second one has 50 nodes with activation sigmoid.
The output layer has 2 nodes, the two classes- Crime and normal, has an activation softmax. 

About Dataset:-
The dataset this project used is a collection of 40 crime videos and 30 normal videos. We collected crime videos from different sources. They are mostly CCTV photages. 

Data preparation:-
Each video is converted into frames and are saved in distinct folders. Then we have manually selected 40 frames from each video which depicts crimes happening. The 40 frames of each video are saved to numbered folders.
You can download the prepared dataset from my google drive by going to this link:
https://drive.google.com/drive/folders/1kXsmOcZyQXyNCuWEHWUphQ7qjo1CANUn?usp=sharing

Methodology:-
For transfer learning we only used the output from the last pooling layer of Inception v3. The code for this part is given in the given file Generate_weights.py where we have passed each frame into the inception-v3 network and extracted a 2048 vector. 
The 2048 vector for each image is extracted in the form of list. We have 40 crime videos, hence 40*40 i.e. 1600 2048-vectors for crime videos and 40*30 i.e 1200 2048-vectors for normal videos. We took 40 frames for each video and grouped them to represent them as a sequence for our RNN. The final list has a length of 2800(1600+1200). We convert it to a 3D array of dimension (70,40,2048) in the beginning of the file train.py. We also created a y variable for storing 1 for crime videos and 0 for normal videos.

Then we split our dataset of 3D array into 80% training data and 20% test data. The model is then trained on this 3D array. See the code in the file train.py.

We achieved an accuracy of 92.86%. You can improve this by trying more hyperparameter tuning and changing epochs and batch_size. You can also make this project real-time by using OpenCV.

Thankyou, hope you undertood this project well.😁😁😁

