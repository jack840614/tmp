from skimage import io, transform
import cv2
import numpy as np
import os
import tensorflow as tf

testing_h = 28
testing_w = 28
testing_path = "./data/testing/"
classes = 9

def read_img(path):
    imgs = []
    labels = []
    for i in range(classes-1):
        f = os.listdir(path+str(i))
        print(path + str(i))
        for im in f:
            img = io.imread(path+str(i)+"/"+im)
            img = transform.resize(img, (testing_h, testing_w,1))
            imgs.append(img)
            labels.append(i)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

data, label = read_img(testing_path)

print(data.shape)
print(label.shape)

# 載入網路

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
saver = tf.train.import_meta_graph('./CNN_net/save_net.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('./CNN_net/'))

graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")

output = graph.get_tensor_by_name("output_eval:0")
# 進行預測

correct = 0
for i in range(len(data)):
    test_output = sess.run(output, feed_dict={x: [data[i]]})
    pred_y = np.argmax(test_output, 1)
    # print(pred_y,label[0])
    if pred_y == label[i]:
        correct += 1


print(correct / len(data))