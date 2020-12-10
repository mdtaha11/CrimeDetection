import cv2
import os

#Extracting all the frames from the video and storin
for dirname, _, filenames in os.walk(r'D:\NORmalll\Normal'):
    for filename in filenames:
        vidcap = cv2.VideoCapture(os.path.join(dirname,filename))
        os.makedirs('D:/'+filename)
        def getFrame(sec):
            vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
            hasFrames,image = vidcap.read()
            if hasFrames:
                
                cv2.imwrite('D:/'+filename+'/'+str(count)+".jpg", image)     # save frame as JPG file
            return hasFrames
        sec = 0
        frameRate = 0.6 #//it will capture image in each 0.5 second
        count=1
        success = getFrame(sec)
        while success:
            count = count + 1
            sec = sec + frameRate
            sec = round(sec, 2)
            success = getFrame(sec)


