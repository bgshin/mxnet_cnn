# Keras for mxNet bug report

## Environment
* Local Server
	* CPU: Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz (48 cores)
	* Mem: 260G
	* GPU: TITAN X (Pascal)


## Issue

* keras\_shape and backend\_shape doesn't match after applying shape changing operations.
	* When performing an tensor operation (like multiplication), keras tries to check the validity of a given operation using keras_shape.
	* If the shape of a tensor is changed by some operation (eg. slicing a tensor), then both of the keras\_shape and backend\_shape should be changed.
	* However, for mxNet, only backend_shape is changed.
	* This causes an error when applying an operations that requires correct shape information.

## Simple example that cuases an error
* File location

```
{project}/src/mx_keras_bug.py
```

* This is the example file (mx\_keras\_bug.py)

```python
import os
# os.environ['KERAS_BACKEND']='mxnet'
os.environ['KERAS_BACKEND']='tensorflow'

from keras.layers import Input, Lambda, merge
from keras.models import Model
import numpy as np

i1 = np.array([[2,3],[2,3],[2,3],[2,3],[2,3]])
i2 = np.array([3,3,3,3,3])

input1_shape = (2,)
input2_shape = (1,)
model_input1 = Input(shape=input1_shape)
model_input2 = Input(shape=input2_shape)
z = model_input1

print 'keras shape = ', z._keras_shape
print 'shape = ', z.shape

z = Lambda(lambda x: x[:,1:2])(model_input1)

print 'keras shape = ', z._keras_shape
print 'shape = ', z.shape

output = merge([z, model_input2], mode='mul')

model = Model([model_input1, model_input2], output)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

print model.predict([i1, i2])
```

* We can choose the backend to either mxnet or tensorflow (line 2,3)
* The expected output is [9,9,9,9,9] - simple multiplication
* Problemetic Line: slicing the first input using the Lambda layer (Line 21)
	* Shapes **before** applying the lambda layer (Line 18,19)
	
	| - | mxNet | Tensorflow |
	| ------ | ------ | ------ |
	| keras_shape |  (None, 2) | (None, 2) |
	| backend_shape | (0L, 2L) | (?, 2) |

	* Shapes **after** applying the lambda layer (Line 23,24)
	
	| - | mxNet | Tensorflow |
	| ------ | ------ | ------ |
	| keras_shape | **(None, 2)** | (None, 1) |
	| backend_shape | (0L, 1L) | (?, 1) |
	

	* keras_shape (for mxnet) should be (None, 1), but it is (None, 2)

	
## Contact

* Bonggun Shin (bonggun.shin@emory.edu)

