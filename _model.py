# coding: utf-8
from __future__ import print_function
#from read_utils import TextConverter
#import codecs
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from skimage import draw,data
import matplotlib.pyplot as plt
import math
import time
import os
import win_unicode_console
win_unicode_console.enable()

class DrawLSTM:
    def __init__(self, is_train=True, num_classes=361, num_pics=32, widths_size=50,
                 lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, train_keep_prob=0.5, use_embedding=True, embedding_size=361, timestep = 26):
        
        #num_seqs, num_steps = num_seqs, num_steps

        self.num_classes = num_classes
        self.num_pics = num_pics
        self.widths_size = widths_size
        self.timestep = timestep
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size

        tf.reset_default_graph()
        self.build_lstm()
        if is_train:
            self.build_loss()
            self.build_optimizer()
        self.saver = tf.train.Saver()

    def build_lstm(self):
        # 创建单个cell并堆叠多层
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.float32, shape=(
                self.num_pics, self.widths_size, self.widths_size), name='inputs')
            self.targets = tf.placeholder(tf.int32, shape=(
                self.num_pics, self.timestep), name='targets')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            net_inputs = self.inputs[:,:,:,np.newaxis]

            feature_network = tl.layers.InputLayer(net_inputs, name='input')
            feature_network = tl.layers.Conv2d(feature_network, 8, (5,5), (1,1), act=tf.nn.relu, name='conv1')
            feature_network = tl.layers.Conv2d(feature_network, 4, (3,3), (1,1), act=tf.nn.relu, name='conv2')
            feature_network = tl.layers.Conv2d(feature_network, 1, (3,3), (1,1), act=tf.nn.relu, name='conv3')
            net_inputs = tf.reshape(feature_network.outputs, [-1, self.widths_size**2])

            if self.use_embedding is False:
                self.lstm_inputs = net_inputs #tf.one_hot(self.inputs, self.num_classes)
            else:
                #with tf.device("/cpu:0"):
                embedding = tf.get_variable('embedding', [self.widths_size**2, self.embedding_size])
                _inputs = tf.matmul(net_inputs, embedding) #tf.nn.embedding_lookup(embedding, self.inputs)
                #Lstm_input = tf.get_variable('Lstm_input', [self.num_pics, self.timestep, self.embedding_size])
                Lstm_input = []
                for _ in range(self.timestep):
                    Lstm_input.append(_inputs)
                #Lstm_input = np.array(Lstm_input)
                #Lstm_input = Lstm_input[np.newaxis]
                Lstm_input = tf.transpose(Lstm_input, [1, 0, 2])
                self.lstm_inputs = Lstm_input



        def get_a_cell(lstm_size, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop

        with tf.name_scope('lstm'):
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)]
            )
            self.initial_state = cell.zero_state(self.num_pics, tf.float32)

            # 通过dynamic_rnn对cell展开时间维度 
            # self.lstm_outputs shaped:  [batch_size, max_time, cell.output_size]`.  
            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.lstm_inputs, initial_state=self.initial_state)

            # 通过lstm_outputs得到概率
            seq_output = tf.concat(self.lstm_outputs, 1)
            x = tf.reshape(seq_output, [-1, self.lstm_size])

            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_classes], stddev=0.1))
                softmax_b = tf.Variable(tf.zeros(self.num_classes))

            self.logits = tf.matmul(x, softmax_w) + softmax_b
            self.logits = tf.reshape(self.logits, [self.num_pics, self.timestep, self.num_classes])
            self.proba_prediction = tf.nn.softmax(self.logits, name='predictions')
            self.proba_prediction = tf.reshape(self.proba_prediction, [self.num_pics, self.timestep, self.num_classes])

    def build_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.targets, self.num_classes)
            y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)
            self.loss = tf.reduce_mean(loss)

    def build_optimizer(self):
        # 使用clipping gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))

    def train(self, batch_generator, max_steps, save_path, save_every_n, log_every_n):
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            # Train network
            step = 0
            new_state = sess.run(self.initial_state)
            for x, y in batch_generator:
                step += 1
                if step % 1000 == 0:
                    if self.learning_rate > 0.00005:
                        self.learning_rate = self.learning_rate * (0.975 ** (step // 1000))
                start = time.time()
                feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: self.train_keep_prob,
                        self.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([self.loss,
                                                     self.final_state,
                                                     self.optimizer],
                                                    feed_dict=feed)

                end = time.time()
                # control the print lines
                if step % log_every_n == 0:
                    print('step/max_steps = {}/{} '.format(step, max_steps),
                          'loss = {:.3f} '.format(batch_loss),
                          'learning_rate = {:f} '.format(self.learning_rate),
                          'process_speed = {:.3f} sec/batch'.format((end - start)))
                if (step % save_every_n == 0):
                    self.saver.save(sess, save_path, global_step=step)
                if step >= max_steps:
                    break
            self.saver.save(sess, save_path, global_step=step)

    def draw_line(self, img, angle):
        length_ = 20
        startpointx = 250
        startpointy = 250
        endpointx = startpointx + int(length_*math.cos(angle[1] / 180.0 * 3.14159))
        endpointy = startpointy + int(length_*math.sin(angle[1] / 180.0 * 3.14159))
        for i in range(self.timestep - 2):
            rr, cc =draw.line(startpointx, startpointy, endpointx, endpointy) 
            img[rr, cc] = 1
            startpointx = endpointx
            startpointy = endpointy
            endpointx = startpointx + int(length_*math.cos(angle[i + 2] / 180.0 * 3.14159))
            endpointy = startpointy + int(length_*math.sin(angle[i + 2] / 180.0 * 3.14159))
        return img

    def test(self, batch_generator):
        
        sess = self.session
        new_state = sess.run(self.initial_state)
        #preds = np.ones((vocab_size, ))  # for prime=[]
        for x, y in batch_generator:
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)
            break
        _results = []
        for i in range(self.num_pics):
            results = []
            for j in range(self.timestep):
                result = np.argmax(preds[i][j]) + 1
                results.append(result)
            _results.append(results)
        #preds shape [self.num_pics, self.timestep, self.num_classes]
        f = open('C:\\Users\\brucezhcw\\Desktop\\test2.txt', 'w')
        for i in range(self.timestep):
            f.write(str(_results[0][i]) + '\n')
        f.close()

        img_label = np.zeros((500, 500), float)
        img_result = np.zeros((500, 500), float)
        img_result = self.draw_line(img_result, _results[0])
        img_label = self.draw_line(img_label, y[0])
        
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.imshow(x[0])
        plt.subplot(3, 1, 2)
        plt.imshow(img_label)
        plt.subplot(3, 1, 3)
        plt.imshow(img_result)
        plt.show()
        #plt.savefig('C:\\Users\\brucezhcw\\Desktop\\test2.png')

    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))