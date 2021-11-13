import cv2
from numpy.typing import _128Bit
from detection_functions import DlibFaceAligner
import numpy as np
import collections

# define a video capture object
vid = cv2.VideoCapture(0)

#Image parameters that inputing in FaceDetector
depressed_width = 240
depressed_height = 180

aligner = DlibFaceAligner('models/shape_predictor_68_face_landmarks.dat',
                          'models/frozen_inference_graph.pb',
                          SSD_input_w = depressed_width,
                          SSD_input_h = depressed_height)

#Create circular buffer with 25 len (by FPS) 
d = collections.deque(maxlen=25)

while(True):
      
    ret, frame = vid.read()
  
    h,w,c = frame.shape

    flag, bottom_lip, top_lip = aligner.get_lips_points(frame)

    if flag:
        diff = abs(bottom_lip - top_lip)
        d.append(diff)
        
        cv2.line(frame,(0, bottom_lip), (w,bottom_lip),(0,0,0),2)
        cv2.line(frame,(0, top_lip), (w,top_lip),(0,0,0),2)
        cv2.putText(frame, "Difference between points: "+str(len(d)), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        
        avg = round(sum(d)/len(d))
        cv2.putText(frame, "AVG for last 25 frames: "+str(avg), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

        if avg > 2:
            cv2.putText(frame, "You're breating with mouth", (0, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    cv2.imshow("Output", frame)

    #Q for exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
vid.release()
cv2.destroyAllWindows()