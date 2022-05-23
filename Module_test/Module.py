import sys
import mediapipe as mp
import tensorflow as tf
import cv2
import pyautogui as pag
from numpy import zeros,expand_dims,argmax,arccos,einsum,degrees,concatenate,linalg,newaxis,array,float32

model_path = 'models/demo_model20.h5'
model = tf.keras.models.load_model(model_path)


def degree(joint): 
    
    v = vector(joint)
    
    angle = arccos(einsum('nt,nt->n', 
            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

    angle = degrees(angle) # Convert radian to degree
    #line 45 ~ 57 손가락 관절들의 각도를 계산하는 코드

    degree = concatenate([joint.flatten(), angle])
    #계산된 각도를 가져옴
    
    return degree


def vector (joint): # Compute angles between joints
    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
    v = v2 - v1 # [20, 3]
    # Normalize v
            
    v = v / linalg.norm(v, axis=1)[:, newaxis]

    return v

def What_Gesture(this_action,last_action):
    if last_action != this_action:
        if this_action == 'F1':
            pag.press('playpause')
        elif this_action == 'F2':
            pag.press('playpause')
        elif this_action == 'F3':
            pag.press('volumeup')
        elif this_action == 'F4':
            pag.press('volumedown')
        elif this_action == 'F5':
            pag.press('volumemute')
        #elif this_action == 'Forward':
            #pag.press('right')
        #elif this_action == 'Rewind':
            #pag.press('left')
    
    last_action = this_action

    return last_action


# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def Set_Video(img):
    img = cv2.flip(img, 1) #카메라 좌우반전을 위해서 flip
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return result,img

def Predict(seq,seq_length):
    input_data = expand_dims(array(seq[-seq_length:], dtype=float32), axis=0)
    #np.expand_dims = 배열의 차원을 추가
    y_pred = model.predict(input_data).squeeze()
    #예측 결과 저장

    i_pred = int(argmax(y_pred))
    #예측된 y값의 최대값의 인덱스를 넣어줌
    conf = y_pred[i_pred]
    #conf에 일치하는 y값을 넣어줌

    return i_pred,conf



