import sys
import cv2
import mediapipe as mp
import numpy as np
import pyautogui as pag
import tensorflow as tf

actions = ['F1','F2','F3','F4','F5']
seq_length = 30

model = tf.keras.models.load_model('models/demo_model10.h5')

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

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1) #카메라 좌우반전을 위해서 flip
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
            #line 45 ~ 57 손가락 관절들의 각도를 계산하는 코드

            d = np.concatenate([joint.flatten(), angle])
            #계산된 각도를 가져옴

            seq.append(d)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            #np.expand_dims = 배열의 차원을 추가
            y_pred = model.predict(input_data).squeeze()
            #예측 결과 저장

            i_pred = int(np.argmax(y_pred))
            #예측된 y값의 최대값의 인덱스를 넣어줌
            conf = y_pred[i_pred]
            #conf에 일치하는 y값을 넣어줌

            if conf < 0.9:
                continue
            #90%이하의 확률이면 정확하지 않은 동작으로 인식

            action = actions[i_pred]
            #예측된 y의 최대값의 인덱스가 예측된 action으로 들어감
            action_seq.append(action)
            #동작을 비교하기 위해서 action_seq라는 배열에 넣기
            
            if len(action_seq) < 3:
                continue
            #영상으로 수집한 데이터 부족으로 다시 실행

            this_action = '?'
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                #마지막 3동작 비교해서 this_action setting
                this_action = action
            
                #if last_action == 'StartMotion' and this_action != 'StartMotion' :
                #시작동작으로 구분하여 실행
                
                if last_action != this_action:
                    if last_action == 'F1' and this_action == 'F2':
                        pag.press('playpause')
                    elif last_action == 'F2' and this_action == 'F1':
                        pag.press('playpause')
                    elif last_action == 'F4' and this_action == 'F5':
                        pag.press('volumeup')
                    elif last_action == 'F5' and this_action == 'F4':
                        pag.press('volumedown')
                    elif last_action == 'F5' and this_action == 'F1':
                        pag.press('volumemute')
                    elif last_action == 'F2' and this_action == 'F3':
                        pag.press('right')
                    elif last_action == 'F3' and this_action == 'F2':
                        pag.press('left')
                    
                
                last_action = this_action

            cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    cv2.imshow('Motion Gestures', img)
    if cv2.waitKey(1) == ord('q'):
        break
