from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import statistics
import collections

def get_key(val):
    for key, value in trial.items():
         if val == value:
             return key

face_classifier = cv2.CascadeClassifier(r'D:\interships_projects\affect_recognition\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')
classifier =load_model(r'D:\interships_projects\affect_recognition\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\model.h5')

emotion_labels = ['Frustrated','Distracted','Confused','Interested','Neutral', 'Bored', 'Surprise']
dict1 = {'Frustrated':0,'Distracted':1,'Confused':2,'Interested':3,'Neutral':4, 'Bored':5, 'Surprise':6}


cap = cv2.VideoCapture(0)

list_labels = []
pie_labels = []
mode_labels = []
time = []
freq = {}
x = 0
limit = 0
flag = 0
trial = {}
pie_prev_max = 4
pie_curr_max = 4
pie_comment = ""
while True:     
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            list_labels.append(dict1.get(label))
            pie_labels.append(label)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    if len(pie_labels)%100==0 and len(pie_labels)!=0:
        frequency = collections.Counter(pie_labels)
        freq = dict(frequency)
        for x1 in freq.keys():
            if x1 not in trial:
                trial[x1] = freq.get(x1)*100.0//len(pie_labels)
        sort_list = list(trial.values())
        sort_list.sort()
        if dict1[get_key(sort_list[-1])]!=4:
            val = dict1[get_key(sort_list[-1])]
        else:
            val = dict1[get_key(sort_list[-2])] 
        pie_curr_max = val
        if pie_curr_max==3 or pie_curr_max==4 or pie_curr_max==6:
            if pie_prev_max==3 or pie_prev_max==4 or pie_prev_max==6:
                pie_comment = "Keep up the interest levels!"
            else:
                pie_comment = "You are understanding the concepts better now. Keep it up!" 
        else:
            if pie_prev_max==3 or pie_prev_max==4 or pie_prev_max==6:
                pie_comment = "Your concentration levels are dipping, try private chatting with the teacher."
            else:
                pie_comment = "Looks like you are facing some trouble understanding the concept. Head over to QnA area to clarify your doubts" 
        f = open("D:/interships_projects/affect_recognition/Emotion_Detection_CNN-main/Prototype/p_analysis.txt","w")   
        f.write(pie_comment)
        f.write("\n")
        f.close()
        pie_prev_max = pie_curr_max

        y = np.array(list(trial.values()))
        mylabels = list(trial.keys())
        plt.pie(y, labels = mylabels)
        plt.savefig("D:/interships_projects/affect_recognition/Emotion_Detection_CNN-main/Prototype/Images/pie.png")
        plt.show()         

        for i in range(limit//10,len(list_labels)//10):
            try:
                mode_labels.append(statistics.mode(list_labels[i*10:i*10+10]))
            except:
                mode_labels.append(4)
            time.append(i*10)
        plt.title('Student Analysis')
        plt.xlabel('Time (secs)')
        plt.ylabel('Emotion')
        plt.plot(time,mode_labels)    
        plt.savefig("D:/interships_projects/affect_recognition/Emotion_Detection_CNN-main/Prototype/Images/line.png")
        plt.show()

        #plt.savefig("D:\interships_projects\affect_recognition\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\graph.png")
        limit += 100
    x += 1    
    trial = {}
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()