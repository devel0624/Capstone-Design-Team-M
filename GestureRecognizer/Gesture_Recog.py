import sys
import cv2
import mediapipe as mp
import tensorflow as tf
import pyautogui as pag
from numpy import zeros,expand_dims,argmax,arccos,einsum,degrees,concatenate,linalg,newaxis,array,float32
from time import sleep , time

model_path = 'models/Gestures_model.h5'
model = tf.keras.models.load_model(model_path)

actions = ['StartMotion', 'PlayPause', 'Forward', 'Rewind', 'VolUp', 'VolDn']
seq_length = 10

seq = []
action_seq = []
last_action = ['','','']
this_action = '?'
Max_time = None

cap = cv2.VideoCapture(0)


def Gesture_Cap():
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        result, img = Set_Video(img)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS) # 랜드마크 표시(기능 요소 없음)

                if Sequence_Gesture(joint): # 카메라에 인식되는 동작에 대한 정보
                    continue

                if Gesture_Predict(): # 인식된 동작 정보(seq)를 이용하여, 현재 동작을 예측
                    continue

                cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), 
                            int(res.landmark[0].y * img.shape[0] + 20)), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
                            color=(255, 255, 255), thickness=2)

        cv2.imshow('Motion Gestures', img)
        if cv2.waitKey(1) == ord('q'):
            break

def What_Action(this_action):
    global last_action
    global Max_time
    print("Choose Action")

    
    if this_action == 'PlayPause':
        pag.press('playpause')
        Max_time = None
        
    elif this_action == 'VolUp':
        pag.press('volumeup')

    elif this_action == 'VolDn':
        pag.press('volumedown')
        
    #lif this_action == 'Mute':
    #    pag.press('volumemute')
    #    Max_time = None

    elif this_action == 'Forward':
        pag.press('right')
    elif this_action == 'Rewind':
        pag.press('left')

    if this_action != '?':
        last_action[2] = last_action[1]
        last_action[1] = last_action[0]
        last_action[0] = this_action

    if last_action[0] == last_action[1] and last_action[1] == last_action[2] :
        sleep(0.2)
    




# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)



def Gesture_Predict():
    global Max_time
    global this_action 

    input_data = expand_dims(array(seq[-seq_length:], dtype=float32), axis=0)
    #np.expand_dims = 배열의 차원을 추가
    y_pred = model.predict(input_data).squeeze()
    #예측 결과 저장

    i_pred = int(argmax(y_pred))
    #예측된 y값의 최대값의 인덱스를 넣어줌
    conf = y_pred[i_pred]
    #conf에 일치하는 y값을 넣어줌

    if conf < 0.9:
        this_action = '?'
        return this_action
    #90%이하의 확률이면 정확하지 않은 동작으로 인식

    else:
        action = actions[i_pred]
        #예측된 y의 최대값의 인덱스가 예측된 action으로 들어감
        action_seq.append(action)
        #동작을 비교하기 위해서 action_seq라는 배열에 넣기
        if len(action_seq) < 3:
            return True
            #영상으로 수집한 데이터 부족으로 다시 실행

        

        if action_seq[-1] == action_seq[-2] == action_seq[-3]:
            #마지막 3동작 비교해서 this_action setting
            this_action = action
            
            #"""
            if Max_time == None and this_action == 'StartMotion' :
                Max_time = time() + 10
                print("Cap_start")

            if Max_time != None and Max_time > time() :
                What_Action(this_action)

            elif Max_time != None and Max_time < time() :
                Max_time = None
            #"""   

 

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

def Sequence_Gesture(joint):
       
    seq.append(degree(joint))

    if len(seq) < seq_length:
        return True
    else:
        return False


def Set_Video(img):
    img = cv2.flip(img, 1) #카메라 좌우반전을 위해서 flip
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return result,img
