import sys
import cv2
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import mediapipe as mp
import numpy as np
import pyautogui as pag
import tensorflow as tf
import time

actions = ['StartMotion', 'PlayPause', 'Forward', 'Rewind', 'VolUp', 'VolDn', 'Mute']
seq_length = 30

model = tf.keras.models.load_model('models/Gestures_model.h5')

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

seq = []
action_seq = []
last_action = None
last_action2 = None
last_action3 = None
Max_time = None
re_time = None

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
            v = v2 - v1 # [20, 3]
            # Normalize v
            
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            d = np.concatenate([joint.flatten(), angle])

            seq.append(d)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            y_pred = model.predict(input_data).squeeze()

            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if conf < 0.9:
                continue

            action = actions[i_pred]
            action_seq.append(action)
            
            if len(action_seq) < 3:
                continue

            this_action = '?'
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action
                if this_action == 'StartMotion' and Max_time == None :
                    Max_time = time.time() + 30

                if Max_time != None and Max_time > time.time() :
                    if this_action == 'PlayPause':
                        pag.press('playpause')
                        time.sleep(0.5)
                    elif this_action == 'VolUp':
                        pag.press('volumeup')
                    elif this_action == 'VolDn':
                        pag.press('volumedown')
                    elif this_action == 'Mute':
                        pag.press('volumemute') 
                        time.sleep(0.5)
                    elif this_action == 'Forward':
                        pag.press('right')
                    elif this_action == 'Rewind':
                        pag.press('left')

                    last_action3 = last_action2
                    last_action2 = last_action
                    last_action = this_action
                
                    if last_action == last_action2 and last_action2 == last_action3 :
                        time.sleep(0.2)
                
                elif Max_time != None and Max_time < time.time() :
                    Max_time = None

            cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    cv2.imshow('Motion Gestures', img)
    if cv2.waitKey(1) == ord('q'):
        break
