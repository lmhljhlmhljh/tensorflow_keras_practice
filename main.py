"""
Written by MinHyung Lee
2020.04.08
researcher of NAMZ


API : tf.keras
linear_regression(선형 회귀)

# How to use
python main.py

# Detail
Answer : Y=2X+A (0<=A<=1/3)
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers mo Dense

# 1. dataset 생성
x_train = np.random.random((1000,1))
y_train = x_train * 2 + np.random.random((1000,1)) / 3
x_test = np.random.random((100,1))
y_test = x_test*2 + np.random.random((100,1)) / 3

"""
# 데이터셋 플롯팅
plt.plot(x_train, y_train, 'ro')
plt.plot(x_test, y_test,'bo')
plt.legend(['train','test'], loc='upper left')
plt.show()
"""

# 2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))

# 3. 모델 학습과정 설정하기
model.compile(optimizer='rmsprop',loss='mse')


print('Model compiling has been done')

# 4. 모델 학습시키기
history = model.fit(x_train, y_train, epochs=500, batch_size=64)
w, b = model.get_weights()
print(w, b)

# 5. 학습과정 살펴보기
plt.plot(history.history['loss'])
plt.ylim(0.0,1.5)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train'],loc='upper left')
plt.show()

# 6. 모델 저장하기
now = datetime.now()
model_name = '{}-{}-{}-linear_regression.h5'.format(now.year, now.month, now.day)
model.save(model_name)


