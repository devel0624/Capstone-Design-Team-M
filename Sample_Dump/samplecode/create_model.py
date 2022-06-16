import numpy as np
import os
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#1번 gpu사용
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#텐서플로우 gpu사용

actions = ['F1','F2','F3','F4','F5']

data = np.concatenate([
    np.load('dataset/seq_F1_1651684338.npy'),
    np.load('dataset/seq_F2_1651684338.npy'),
    np.load('dataset/seq_F3_1651684338.npy'),
    np.load('dataset/seq_F4_1651684338.npy'),
    np.load('dataset/seq_F5_1651684338.npy')
], axis=0)

#npy데이터를 합쳐서 하나의 data 배열 생성

data.shape

x_data = data[:, :, :-1]
#마지막 빼고는 다 data셋
labels = data[:, 0, -1]
#마지막 위치는 label

print(x_data.shape)
print(labels.shape)

y_data = tf.keras.utils.to_categorical(labels, num_classes=len(actions))
# to_categorical - action의 갯수만큼 one hot 인코딩
y_data.shape

x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)
#데이터를 32비트 실수형으로 변경

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=2022)
#train_test_split - test set과 validation set 분리
#random_state= 시드값 고정


print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
#쉐입 출력

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, activation='relu', input_shape=x_train.shape[1:3]),
    #x_train.shape[1:3] index 1은 나누는 값 2는 각도값들
    #영상을 잘라서 각도를 계속 계산하기 때문에 값이 많아져서 나눠서 학습시키는 LSTM방식 사용
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(actions), activation='softmax')
])
#sigmoid, relu, softmax
#relu - 은닉층, softmax - 확률값, sigmoid - yse or no

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.summary()

hist = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=200, #200회
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint('models/Gestures_model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        #(파일패스, 발리데이션 정확도 기준, 저장될떄 출력, 모니터기준 최고값 저장, 큰값 작은값 구별 자동)
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')
        #콜백함수 발리데이션 정확도기준, learnig rate 감소비율 0.5, 비율조정
    ]
)

import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()
  
model = tf.keras.models.load_model('models/Gestures_model.h5')

y_pred = model.predict(x_val)

multilabel_confusion_matrix(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))
