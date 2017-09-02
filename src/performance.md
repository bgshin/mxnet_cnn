# Keras Speed Comparison between mxNet and Tensorflow

## Environment
* Local Server
	* CPU: Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz (48 cores)
	* Mem: 260G
	* GPU: TITAN X (Pascal)
* Model
	* Embedding: Custom built 400 dimensional w2v
	* Classifier: CNN based sentiment analysis
* Dataset
	* SST: (5 classes)
	* \# of Training data: 8k


## Speed Comparison

### Train
| Job | mxNet | Tensorflow |
| ------ | ------ | ------ |
| Compiling a Model (+loading w2v) | 12.29 | 11.821 |
| Saving a Model  | 22.25 | 11.823 |
| Training an Epoch  | 6.60 | 4.70 |

* For mxNet, if the first epoch is excluded (17.95s), then the avg time becomes 2.81s.
* Time of loading a model varies a lot for both of the backends.


### Decode
| Job | mxNet | Tensorflow |
| ------ | ------ | ------ |
| Compiling a Model (+loading a trained model) | 20.27 | 18.55 |
| '-- Loading a Trained Model | 20.21 | 18.33 |
| Decoding 8k Samples | 18.11 | 1.57 |

* Time of Loading a model varies a lot for both of the backends.
* When decoding using mxNet, actual decoding time seems to be short (less than 2 secs). However, for some reason, the function call of "model.evaluate" lasted up to 50 secs.


## Log data

* Please refer the following log files for more details

```
{project}/src/train_speed_mxnet.log
{project}/src/train_speed_tf.log
{project}/src/decode_speed_mxnet.log
{project}/src/decode_speed_tf.log
```

## Contact

* Bonggun Shin (bonggun.shin@emory.edu)