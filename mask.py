import cv2
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
import os

model=load_model('model.h5')

img_width, img_height= 150,150

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#HaarcascadeCalssifier is a pretrained classifier to detect faces on the image

cap=cv2.VideoCapture(0) #provide path for video file

img_count_full=0

font=cv2.FONT_HERSHEY_SIMPLEX #font style
org=(1,1)                       #origin
class_label='' 
fontScale=1                     #font size
color=(255,0,0)                 #colour for text
thickness=2                     #thickness of text


while True:
    img_count_full+=1
    response,color_img=cap.read()  #to capture the image
    
   
    #response will be either True or False
    #we will get image or a frame in color_img
    
    if response==False:
        break
        
    scale=50
    width=int(color_img.shape[1]*scale/100)
    height=int(color_img.shape[0]*scale/100)
    dim=(width,height)
    #if image has higher resolution model cannot predict correctly and system power will be required at a higher rate
    #so we reduce size by rescaling by 50 percent of original image 
  
    gray_img=cv2.cvtColor(color_img,cv2.COLOR_BGR2GRAY) 
    #convert imgae to grayscale as cascadeclassifier model works only on grayscale images
    
    faces=face_cascade.detectMultiScale(gray_img,1.4,6)
    
    #1.1 is scaling factor so the image is rescaled to smaller extent
    #6 is min neighbors basically in an image how many face is to be detected
    #it will detect all the faces present and return the coordinate values (x,y,w,h)
    
    img_count=0  #when we save an image we save with different names 
    for (x,y,w,h) in faces:
        org=(x-10,y-10) #for writing text 10 pixels above the rectangle
        img_count+=1
        

        color_face=color_img[y:y+h,x:x+w] #extracting the face portion from the image
        cv2.imwrite('input/%d%dface.jpg'%(img_count_full,img_count),color_face)
        img=load_img('input/%d%dface.jpg'%(img_count_full,img_count),target_size=(img_width,img_height))
       

        
      

        #load stored image with desired target size
        
        
        img=img_to_array(img)   #convert to array
        img=np.expand_dims(img,axis=0)   #change dimension of image
        prediction= model.predict(img)
    
        if prediction==0:
            class_label="Mask"
            color=(0,255,0)
            
        else:
            class_label="No Mask"
            color=(255,0,0)
        
        print(class_label)
        
        cv2.rectangle(color_img,(x,y),(x+w,y+h),(0,0,255),3)
        
        #thickness is 3
        #cv2.rectangle(image,start point,end point,color,thickness)
        
        cv2.putText(color_img,class_label,org,font,fontScale,color,thickness,cv2.LINE_AA)
        
        #cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
        
        cv2.imshow("Face mask detection",color_img)
        os.remove('input/%d%dface.jpg'%(img_count_full,img_count))
    
    if cv2.waitKey(1) == ord('q'):
        break
            
cap.release()
cv2.destroyAllWindows()



        
    