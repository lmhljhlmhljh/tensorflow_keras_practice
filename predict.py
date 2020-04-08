"""
Written by MinHyung Lee
2020.04.08
researcher of NAMZ

API : tf.keras
linear_regression(선형 회귀)

# How to use
python predict.py [model_name] [input]

model_name = *.h5
input => type==int or type==float

#Detail
Answer : Y=2X+A (0<=A<=1/3)
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
	model = load_model(args.model_name)
	model_input = np.array([[args.input]])
	print(model.predict(model_input))