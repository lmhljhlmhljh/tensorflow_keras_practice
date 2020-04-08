"""
Written by MinHyung Lee
2020.04.08
researcher of NAMZ


API : Keras
logistic_regression(로지스틱 회귀)

# How to use
python main.py

#Detail
0~55 -> 0
45~100 -> 1
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

Adam = tf.keras.optimizers.Adam

# 1. dataset 생성
X_1 = np.random.randint(55,size=(1000,1))
Y_1 = np.array([0]*1000)
X_2 = np.random.randint(45,100,size=(1000,1))
Y_2 = np.array([1]*1000)

X = np.append(X_1,X_2)
Y = np.append(Y_1,Y_2)

# scikit_learn에서 제공하는 훈련세트와 데이터세트 분리 패키지
x_train,x_test,y_train,y_test = train_test_split(X,Y)

'''
#plotting
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

plt.plot(x_train, y_train,'ro')
plt.plot(x_test, y_test,'bo')
plt.legend(['train','test'], loc='upper left')

plt.show()
'''

# 2. 모델 구성 및 컴파일
model = Sequential()
model.add(Dense(2, input_dim=1))
model.add(Dense(1, activation='relu'))
model.compile(optimizer = Adam(lr=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

# 3. 학습
hist = model.fit(x_train, y_train, epochs=1000, batch_size=64, validation_data=(x_test, y_test))

# 4. 학습과정 살펴보기
plt.plot(hist.history['loss'])
plt.ylim(0.0,1.5)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train'],loc='upper left')
plt.show()

# 5. 모델 저장하기
now = datetime.now()
model_name = '{}-{}-{}-logistic_regression.h5'.format(now.year, now.month, now.day)
model.save(model_name)