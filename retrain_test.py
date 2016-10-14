# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Test the retrained model.



This program creates a graph from a saved GraphDef protocol buffer,
adds a top layer from the retrained model,
and runs inference on a test image set. It outputs the accuarcy of both
each class and the whole image set.
Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import tarfile
import glob
import numpy as np
from six.moves import urllib
import tensorflow as tf
import time

from tensorflow.python.platform import gfile

FLAGS = tf.app.flags.FLAGS

# output_graph.pb:
#   Binary representation of the GraphDef protocol buffer.
# output_labels.txt:
#   Text representation of the label.
# these two files need to be put in the /tmp/imagenet folder
tf.app.flags.DEFINE_string(
    'model_dir', '/tmp/imagenet',
    """path to inception v3 model, """
    """output_graph.pb and """
    """output_labels.txt """)
tf.app.flags.DEFINE_string('image_dir', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long


class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               uid_lookup_path=None):
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'output_labels.txt')
    self.node_lookup = self.load(uid_lookup_path)

  def load(self,uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      uid_lookup_path:integer node ID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)

    # Loads mapping from integer node ID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    uid = 0
    p = re.compile(r'\b\w+\b')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid_to_human[uid] = parsed_items[0]
      uid += 1
    return uid_to_human

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved output_graph.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'output_graph.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image_list):
  """Runs inference on test image set.

  Args:
    A dictionary containing an entry for each label subfolder, with test images in it.

  Returns:
    nothing.
  """
 
  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'final_result:0' A tensor containing the prediction across your own output_graph.pb
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the final_result tensor by feeding the image_data as input to the graph.
    overall_sum = 0;
    overall_accuarcy = 0; 
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    # calculate the accuracy for each class
    # class_sum: numbers of images in the class
    # class_accuarcy: numbers of images which classify correctly in teh class
    # overall_sum: numbers of images in the whole test iamge set
    # overall_accuaracy: numbers of images which classify correctly in the who test image set
    for label, test_image_list in image_list.items():
      class_sum = len(test_image_list)
      class_accuarcy = 0;
      for image in test_image_list:
        image_data = tf.gfile.FastGFile(image, 'rb').read()
        predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        # Creates node ID --> English string lookup.
        node_lookup = NodeLookup()
        top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
        if (node_lookup.id_to_string(top_k[0]) == label):
          class_accuarcy += 1
      overall_sum += class_sum
      overall_accuarcy += class_accuarcy
      print(label + '  number of images: {}   accuarcy: {}'.format(class_sum, class_accuarcy/class_sum) )
    print( 'number of all images: {}   overall accuarcy: {}'.format(overall_sum, overall_accuarcy/overall_sum) )   



def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)



def create_image_lists(image_dir):
  """Builds a list of test images from the file system.

  Analyzes the sub folders in the image directory, describing the lists of images for each label and their paths.

  Args:
    image_dir: String path to a folder containing subfolders of images.

  Returns:
    A dictionary containing an entry for each label subfolder, with images in it.
  """
  if not gfile.Exists(image_dir):
    print("Image directory '" + image_dir + "' not found.")
    return None
  result = {}
  sub_dirs = [x[0] for x in os.walk(image_dir)]
  # The root directory comes first, so skip it.
  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []
    dir_name = os.path.basename(sub_dir)
    if dir_name == image_dir:
      continue
    print("Looking for images in '" + dir_name + "'")
    for extension in extensions:
      file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
      file_list.extend(glob.glob(file_glob))
    if not file_list:
      print('No files found')
      continue
    label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    result[label_name] = file_list
  return result


def main(_):
  maybe_download_and_extract()
  image_list = create_image_lists(FLAGS.image_dir)
  run_inference_on_image(image_list)


if __name__ == '__main__':
  tf.app.run()