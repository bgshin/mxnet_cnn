import os
os.environ['KERAS_BACKEND']='mxnet'
# os.environ['KERAS_BACKEND']='tensorflow'

from keras.layers import Input, Lambda, merge
from keras.models import Model
import numpy as np

i1 = np.array([[2,3],[2,3],[2,3],[2,3],[2,3]])
i2 = np.array([3,3,3,3,3])

input1_shape = (2,)
input2_shape = (1,)
model_input1 = Input(shape=input1_shape)
model_input2 = Input(shape=input2_shape)

# z = Lambda(lambda x: x[:,0:1])(model_input1)
z = Lambda(lambda x: x[:,1:2])(model_input1)

print 'keras shape = ', z._keras_shape
print 'shape = ', z.shape

output = merge([z, model_input2], mode='mul')

model = Model([model_input1, model_input2], output)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

print model.predict([i1, i2])
