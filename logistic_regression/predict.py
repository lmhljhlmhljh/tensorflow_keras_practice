"""
Written by MinHyung Lee
2020.04.08
researcher of NAMZ


API : Keras
logistic_regression(로지스틱 회귀)

# How to use
python predict.py [model_name] [input]

model_name = *.h5
input => type==int or type==float

#Detail
0~55 -> 0
45~100 -> 1
"""

import argparse
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser(description='predict with model')
parser.add_argument('model_name', help='put the name of the model')
parser.add_argument('input', type=float, help='The result should be (input)*2+A ; (0<=A<=1)')
args = parser.parse_args()

if __name__ == '__main__' :
	# model 가져오기
	model = load_model(args.model_name)
	# 숫자를 받아오기 (numpy object로 변환)
	model_input = np.array([[args.input]])

	# predict
	if model.predict(model_input) < 0.5 :
		print(0)
	else :
		print(1)