# image-recognition
Image recognition for dangerous items. TensorFlow + inception v3.


## Algorithm part(tensorflow)
These files let you take the pre-trained inception v3 image recognition model, re-train it on your own data.

### Background reading:

* Basic image recognition in tensorflow: https://www.tensorflow.org/versions/r0.9/tutorials/image_recognition/index.html
* Retraining walk through in tensorflow: https://www.tensorflow.org/versions/r0.9/how_tos/image_retraining/index.html

### Prereqs

* Python scripts: Python, tensorflow

### Script Usage
retrain.py : retrain the model 
retrain_test.py : test the accuray of the retrained model 
classify_image.py : classify the test image

#### tensorflow setup 
* To run the program, first you need to setup tensorflow. You can get help from https://www.tensorflow.org/versions/r0.11/get_started/index.html (We suggest the 'Installing from sources' method)

#### retrain model
* Tensorflow has a pre-trained model called inception for image recognition. After you setup TensorFlow, you can retrain this model to work on your own dataset.

* Once you have the images, you can build the retrainer like this, from the root of your TensorFlow source directory:

```
bazel build tensorflow/examples/image_retraining:retrain
```
* The retrainer can then be run like this (python3.5):

```
python3 {YOU_PATH}/retrain.py --image_dir {DATASET_PATH}
```

* You can add distortion, modify some parameters like this (details in retrain.py):

```
python3 {YOU_PATH}/retrain.py --image_dir {DATASET_PATH} --random_crop 5 --how_many_training_steps=50000
```

* The output model is saved in /tmp, the default names are output_graph.pb && output_label.txt

* You can use retrain_test.py to test the performance of your retrain model.

#### classification
* After you retrain the model, you could use classify_image.py to classify the test image.

* Change the file name to fit your retrained model in classify_image.py:
```
line 75 : uid_lookup_path = os.path.join(FLAGS.model_dir, 'output_labels.txt')
line 111 : with tf.gfile.FastGFile(os.path.join(FLAGS.model_dir, 'output_graph.pb'), 'rb') as f:
```
