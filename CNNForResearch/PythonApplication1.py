import os
import __future__
import cv2
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
NUM_CLASSES = 10
IMG_PXLS=28
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

###この辺りから画像処理
testImgs = []
testImgsLabel = []
"""
for i in range(10):
    img = cv2.imread(str(i)+ "test.jpg")
    img = cv2.resize(img, (IMG_PXLS, IMG_PXLS))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img =1-img.flatten().astype(np.float32)/255
    testImgs.append(img)
    tmp = np.zeros(NUM_CLASSES)
    tmp[i] = 1
    testImgsLabel.append(tmp)
testImgs = np.asarray(testImgs)
testImgsLabel = np.asarray(testImgsLabel)
#print(testImgs)
#print(mnist.test.images)
print(mnist.test.labels)
"""
###ここまで画像処理
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
def weightVariable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
#Tensorを正規分布かつ標準偏差(0.1)の２倍までのランダムな値で初期化する
  return tf.Variable(initial)
def biasVariable(shape):
  initial = tf.constant(0.1, shape=shape)
#初期バイアス0.1
  return tf.Variable(initial)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#縦横1マス間隔でフィルタを移動させる，入力画像と同サイズにフィルタリングされた出力画像がなるように出力画像周辺を0で埋める
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
#2*2フィルタで縦横2マス間隔でフィルタを移動，またゼロパディングであるので画像テンソルが28*28であれば
#画像テンソル（28*28)/フィルタテンソル(2*2)より画像サイズは14*14になる

W_conv1 = weightVariable([5, 5, 1, 32])
#[縦pxl, 横pxl, 入力チャンネル（グレースケールなので1)，出力フィルタ枚数（各ｐxlに32重のフィルタがあるといえる）]
b_conv1 = biasVariable([32])
x_image = tf.reshape(x, [-1,28,28,1])
#入力画像のベクトルxを28*28のグレースケール画像を表すtensorにリシェープ
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#入力画像テンソルを畳み込み，バイアス加算してReLU
h_pool1 = max_pool_2x2(h_conv1)
#それをプーリング，ここで画像サイズは14*14に

W_conv2 = weightVariable([5, 5, 32, 64])
#二層目，一層目の結果に掛けられる重みテンソル．[縦pxl, 横pxl,入力チャンネル数（第一層と同じ，フィルタ枚数), 出力チャネル数]
b_conv2 = biasVariable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#このプーリングで画像サイズは7*7

W_fc1 = weightVariable([7 * 7 * 64, 1024])
#第二層の結果である，7*7の画像が64枚あるテンソルをflattenしてる，出力チャネルが1024
b_fc1 = biasVariable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
#第二層の出力テンソルをflatten
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#flattenされた出力テンソルに重みをかけバイアスを加算，それをReLU
###以下で過学習を防ぐため結果を一部落としてる
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weightVariable([1024, 10])
#入力チャンネル1024,出力チャネル10(結果に対応)
b_fc2 = biasVariable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#同様にして重みつけしたテンソルをバイアスドしそれをSoftmax回帰，これが結果



cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
for i in range(10000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
      #a
      #eee

    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
 
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
saver.save(sess, ".\ckpt\model.ckpt")

