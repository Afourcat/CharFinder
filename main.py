#! /usr/bin/python
import sys
import numpy as np
import tensorflow as tf
import binascii
from random import randrange

nb_answers = 10
nb_test = 300
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def main():
    """Example of a tensor flow program that find a char from a byte"""
    x = tf.placeholder(tf.float32, [None, 8])
    W = tf.Variable(tf.zeros([8, 26]))
    b = tf.Variable(tf.zeros([26]))

# place older of result
    y = tf.nn.softmax(tf.matmul(x, W) + b)

# place holder of correct answers
    y_ = tf.placeholder(tf.float32, [None, 26])

# cross entropy algortihm
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# train algorithm
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)

    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    for _ in range(nb_test):
        batch_x, batch_y = get_training()
        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

#test trained model
    #prediction test
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    #accuracy value
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_x, test_y = get_training()
    print(sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))

def get_training():
    batch_x = []
    batch_y = []
    for i in range(nb_answers):
        string = "011"
        for i in range(5):
            string = string + str(randrange(0, 2))
        n = int(string, 2)
        char = binascii.unhexlify('%x' % n)
        x = []
        for i in range(8):
            x.append(string[i])
        letter = ''
        for letter in alphabet:
            if (str(letter) == str(char)):
                break
        y = []
        for i in range(26):
            if (str(alphabet[i]) == str(letter)):
                y.append(1)
            else:
                y.append(0)
        batch_x.append(x)
        batch_y.append(y)
    return batch_x, batch_y

if __name__ == '__main__':
    main()
