import sys
import cv2
import Module as mod1
from numpy import zeros
from Module import mp_hands, mp_drawing

actions = ['F1','F2','F3','F4','F5']
seq_length = 30




cap = cv2.VideoCapture(0)

seq = []
action_seq = []
last_action = None

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    result, img = mod1.Set_Video(img)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]


            degree = mod1.degree(joint)
            #계산된 각도를 가져옴

            seq.append(degree)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS) # 랜드마크 표시(기능 요소 없음)

            if len(seq) < seq_length:
                continue

            i_pred, conf = mod1.Predict(seq,seq_length)
            

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
              
                  
            
                last_action = mod1.What_Gesture(this_action, last_action)

            cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    cv2.imshow('Motion Gestures', img)
    if cv2.waitKey(1) == ord('q'):
        break