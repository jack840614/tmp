import tensorflow as tf
import numpy as np
import os
from skimage import io, transform
from model import *

batch_size = 16
training_h = 28
training_w = 28
training_path = "./data/training/"
classes = 9

def read_img(path):
    imgs = []
    labels = []
    for i in range(classes):
        f = os.listdir(path+str(i))
        print(path+str(i))
        for im in f:
            img = io.imread(path+str(i)+"/"+im)
            img = transform.resize(img, (training_h, training_w,1))
            imgs.append(img)
            labels.append(i)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

data, label = read_img(training_path)

print(data.shape)
print(label.shape)

tf_x = tf.placeholder(tf.float32, [None, training_h, training_w, 1], name='x')
tf_y = tf.placeholder(tf.int32, [None, ], name='y_')


net = ResNet50({'data': tf_x}, is_training=False, num_classes=classes)
fc = net.layers['fc']
# print(output.get_shape()[-1].value)
# flat = tf.reshape(output, [-1, output.get_shape()[1].value*output.get_shape()[2].value*output.get_shape()[3].value])
# flat = tf.reshape(output, [-1, output.get_shape()[-1].value])
output = tf.layers.dense(fc, classes)

loss = tf.losses.sparse_softmax_cross_entropy(labels=tf_y, logits=output)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(output, 1), tf.int32), tf_y)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

b = tf.constant(value=1, dtype=tf.float32)
output_eval = tf.multiply(output, b, name='output_eval')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)


# 一次抓batch_size比資料，shuffle決定是否隨機
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


# 訓練
for epoch in range(20):
    print("epoch: ", epoch + 1)
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(data, label, batch_size, shuffle=True):
        _, err, ac = sess.run([train_op, loss, acc], feed_dict={tf_x: x_train_a, tf_y: y_train_a})
        train_loss += err;
        train_acc += ac;
        n_batch += 1
        # print("batch:",n_batch)
    print("train loss: %f" % (train_loss / n_batch))
    print("train acc: %f" % (train_acc / n_batch))

# 存取模型
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./CNN_net" + str(epoch) + "/save_net.ckpt")