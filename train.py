import tensorflow as tf
import numpy as np
import cv2
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import auto_encoder

def single_batch(image_list):
    images = []
    for s in image_list:
        try:
            i = cv2.imread(s)
            j = cv2.resize(i, dsize=(512,512))
            j.resize((1, 512, 512, 3))
            images.append(j)
        except :
            pass

        if len(images) == 10:
            break

    images = np.concatenate(images, axis=0)
    return images

image_dir = '/raid/workspace/zhengdaren/dresden/jpg'
log_dir = './logs'

image_list = list()

batch_size = 10
epoch = 20
keep_prob = 0.9
global_step = 0

image_list = list(os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.JPG') )

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    images_holder = tf.placeholder(tf.float32, [batch_size, 512, 512, 3])
    autoencoder = auto_encoder.AutoEncoder(npy_path='./autoencoder-save.npy')
    autoencoder.build(images_holder)

    cost = tf.reduce_sum((autoencoder.bn6 - autoencoder.bn1) ** 2)
    tf.summary.scalar(name='loss', tensor=cost)
    learning_rate = tf.train.exponential_decay(0.0001, 40000, 400, 0.975)
    train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir+'/train', sess.graph)
    sess.run(tf.global_variables_initializer())

    for i in range(epoch):
        for j in range(int(len(image_list) / batch_size)-1):
            image_batch = image_list[j*batch_size : (j+2)*batch_size]
            images = single_batch(image_batch)
            data_dict = {images_holder:images}
            sess.run(train, feed_dict=data_dict)
            if global_step % 10 ==0:
                summary = sess.run(merged, feed_dict=data_dict)
                train_writer.add_summary(summary, global_step)
            if global_step % 50 == 0:
                print("Global Step {} OK".format(global_step))
            global_step += 1

        autoencoder.save_npy(sess=sess)

    train_writer.close()


