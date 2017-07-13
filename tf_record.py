import tensorflow as tf
import numpy as np
import cv2
import os

'''
Transform the raw image data to the tfrecords file

In tf.train.Example, the feature is a dict which can has 
int list, float list, byte list and so on
'''

image_dir = '/raid/workspace/zhengdaren/dresden/jpg'
batch_size = 100

image_list = list(os.path.join(image_dir,f) for f in os.listdir(image_dir) if f.endswith('.JPG'))

for i in range(int(len(image_list)/batch_size)):
    writer = tf.python_io.TFRecordWriter('./tfrecords/train_{}.tfrecords'.format(i))
    for s in image_list[i*batch_size:(i+1)*batch_size]:
        try:
            img = cv2.imread(s)
            img = cv2.resize(img, (512, 512))
            img_raw = img.tobytes()

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    }
                ))
            writer.write(example.SerializeToString())
        except:
            pass
    writer.close()
    print('train_{}.tfrecords  OK'.format(i))