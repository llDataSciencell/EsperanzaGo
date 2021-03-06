# coding:utf-8
# TODO: batchでまんべんなくデータが含まれるようにする。
# TODO: オブジェクトだからこのファイルを実行しても何も起きないので、mainから実行する。
# sgf読み込みで新しい棋譜になったら盤面を初期化する
import sgf  # Please install "sgf 0.5". You can install it by using the command of "pip install sgf".
import re
import os  # osモジュールのインポート
from game import Player
from game import State
from search import MontecarloSearch
from go import Go
from input_plane import MakeInputPlane
import tensorflow as tf
import math
from go import GoVariable
from go import GoStateObject
from numpy import *
import traceback
# パスはどうする？forwardした結果一番良い答えがパスかもしれない
import sys
import datetime
import numpy as np
import tensorflow as tf

# from tensorflow.python import control_flow_ops
# sysモジュールをリロードする
reload(sys)
# デフォルトの文字コードを変更する
sys.stderr.write(str(sys.setdefaultencoding('utf-8')))


# デフォルトの文字コードを出力する
# from guppy import hpy
# h = hpy()


class FastPolicyNetwork(GoVariable):
    character_list = [chr(i) for i in range(97, 97 + 26)]

    def __init__(self):
        # self.train()
        self.train()

    def reshape_board(self, board_array):
        reshaped_boards = []
        for i in xrange(len(board_array)):
            reshaped_boards.append(reshape(board_array[i], 361))
        return reshaped_boards

    def reshape_answer_board(self, board_array):
        return reshape(board_array, 361)

    def invert_board_input(self, board_array):
        for i in xrange(len(board_array)):
            board_array[i] = board_array[i][::-1]
        return board_array

    def invert_board_answer(self, board_array):
        board_array[::-1]
        return board_array

    def rotate90_input(self, board_array):
        for i in xrange(len(board_array)):
            board_array[i] = rot90(board_array[i])
        return board_array

    def rotate90_answer(self, board_array):
        # 90度回転させるために、配列を２次元にした方が良い。input shapeもNone,1,361にする。
        rot90(board_array)
        return board_array



    def get_particular_variables(self, name):
        return {v.name: v for v in tf.global_variables() if v.name.find(name) >= 0}


    def train(self):
        players = [Player(0.0, 'human'), Player(1.0, 'human')]
        players[0].next_player = players[1]
        players[1].next_player = players[0]
        # player = players[0]
        rule = Go()

        print "starting tf.device(/gpu:0)"
        sess = tf.InteractiveSession()
        # with tf.device("/gpu:1"):

        n_out = 95
        with tf.device("/cpu:0"):
            def weight_variable(shape):
                """適度にノイズを含んだ（対称性の除去と勾配ゼロ防止のため）重み行列作成関数
                """

                initial = tf.truncated_normal(shape, stddev=0.05)
                return tf.Variable(initial)

            def bias_variable(shape):
                """バイアス行列作成関数
                """
                initial = tf.constant(0.1, shape=shape)
                return tf.Variable(initial)

            def conv2d(x, W):
                """2次元畳み込み関数
                """
                return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

            def batch_normalization(shape, input):
                eps = 1e-5
                gamma = weight_variable([shape])
                beta = weight_variable([shape])
                mean, variance = tf.nn.moments(input, [0])
                return gamma * (input - mean) / tf.sqrt(variance + eps) + beta


            with tf.variable_scope('fast'):
                phase_train = tf.placeholder(tf.bool, name='phase_train')
                # 5290000=23*23*100*100  88200=21*21*
                # データ用可変2階テンソルを用意
                x_input = tf.placeholder("float", shape=[None, 8, 361])
                # 正解用可変2階テンソルを用意
                y_ = tf.placeholder("float", shape=[None, 361])

                # 画像を2次元配列にリシェイプ 第1引数は画像数(-1は元サイズを保存するように自動計算)、縦x横、チャネル
                #x_image = tf.reshape(x_input, [-1, 19, 19, 8])
                x_input_flat = tf.reshape(x_input, [-1, 19 * 19 * 8])
                weight_fully_connected1 = weight_variable([19 * 19 * 8,19*19+100])
                bias_fc1 = weight_variable([19*19+100])
                hidden_fully_connect1 = tf.nn.relu(tf.matmul(x_input_flat, weight_fully_connected1) + bias_fc1)
                #bn_hidden1=batch_normalization(19*19+100,hidden_fully_connect1)
                weight_fully_connected2 = weight_variable([19*19+100, 361])
                bias_fc2 = bias_variable([361])
                y_conv = tf.nn.softmax(tf.matmul(hidden_fully_connect1, weight_fully_connected2) + bias_fc2)

        cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
        # cross_entropy=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,labels=y_))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(self.get_particular_variables('fast'))
        # 指定したパス内の全てのファイルとディレクトリを要素とするリストを返す
        # 自分の環境でのsgf_filesへのパスを書く。
        print "y_!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        print y_

        tf.summary.scalar("cross_entropy", cross_entropy)
        tf.summary.scalar("accuracy", accuracy)
        # tf.summary.scalar("accuracy",accuracy)
        # tf.summary.scalar("correct_prediction",correct_prediction)
        # tf.summary.scalar("train_step",train_step)
        # merged = tf.summary.merge_all()
        files = os.listdir(os.getcwd() + "/kifu")
        print "kifu loaded!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        init = tf.global_variables_initializer()
        xTrain = []
        yTrain = []

        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('tensorflow_logs/train', sess.graph)
            # test_writer  = tf.summary.FileWriter('tensorflow_logs/test',sess.graph)
            sess.run(init)  # If it is first time of learning

            num = 0

            batch_count_num = 0
            train_count_num = 0

            # saver.restore(sess, './Network_Backup/fast_policy')

            make_input = MakeInputPlane()

            step = 0
            ckpt_num = 100
            batch_count_sum_all = 0
            for _ in xrange(100):
                print "batch_count_sum_all=0 passed"
                continue_kifu_num = 0

                for file_name in files:

                    # print h.heap()
                    continue_kifu_num += 1
                    if continue_kifu_num < 10:
                        continue

                    step += 1
                    '''
                    if continue_kifu_num % 100 == 0:
                        result = sess.run([merged,cross_entropy], feed_dict=feed_dict(False))
                        summary_str=result[0]
                        acc = result[1]
                        train_writer.add_summary(summary_str,step)
                    '''
                    with open("kifu/" + file_name) as f:
                        try:
                            collection = sgf.parse(f.read())
                            flag = False
                        except:
                            continue
                        try:
                            go_state_obj = GoStateObject()

                            # print "通過"
                            for game in collection:
                                for node in game:
                                    if flag == False:
                                        flag = True
                                        continue
                                    lists = node.properties.values()
                                    # print lists
                                    internal_lists = lists[0]
                                    position = internal_lists[0]
                                    xpos = self.character_list.index(position[0])
                                    ypos = self.character_list.index(position[1])
                                    pos_tuple = (xpos, ypos)
                                    # print xpos,ypos
                                    if node.properties.has_key('B') == True:
                                        current_player = players[0]
                                    elif node.properties.has_key('W') == True:
                                        current_player = players[1]
                                    # print "move ends"

                                    num += 1

                                    if num > 80:
                                        #print "board adding"
                                        input_board = make_input.generate_input(go_state_obj, current_player)

                                        answer_board = make_input.generate_answer(pos_tuple)
                                        xTrain.append(self.reshape_board(input_board))
                                        yTrain.append(self.reshape_answer_board(answer_board))
                                        # print self.reshape_answer_board(answer_board)
                                        # print self.reshape_answer_board(answer_board)
                                        input_board2 = self.rotate90_input(input_board)
                                        answer_board2 = self.rotate90_answer(answer_board)
                                        xTrain.append(self.reshape_board(input_board2))
                                        yTrain.append(self.reshape_answer_board(answer_board2))

                                        input_board3 = self.rotate90_input(input_board2)
                                        answer_board3 = self.rotate90_answer(answer_board2)
                                        xTrain.append(self.reshape_board(input_board3))
                                        yTrain.append(self.reshape_answer_board(answer_board3))

                                        input_board4 = self.rotate90_input(input_board3)
                                        answer_board4 = self.rotate90_answer(answer_board3)
                                        xTrain.append(self.reshape_board(input_board4))
                                        yTrain.append(self.reshape_answer_board(answer_board4))

                                        input_board5 = self.invert_board_input(input_board4)
                                        answer_board5 = self.invert_board_answer(answer_board4)
                                        xTrain.append(self.reshape_board(input_board5))
                                        yTrain.append(self.reshape_answer_board(answer_board5))

                                        input_board6 = self.rotate90_input(input_board5)
                                        answer_board6 = self.rotate90_answer(answer_board5)
                                        xTrain.append(self.reshape_board(input_board6))
                                        yTrain.append(self.reshape_answer_board(answer_board6))

                                        input_board7 = self.rotate90_input(input_board6)
                                        answer_board7 = self.rotate90_answer(answer_board6)
                                        xTrain.append(self.reshape_board(input_board7))
                                        yTrain.append(self.reshape_answer_board(answer_board7))

                                        input_board8 = self.rotate90_input(input_board7)
                                        answer_board8 = self.rotate90_answer(answer_board7)
                                        xTrain.append(self.reshape_board(input_board8))
                                        yTrain.append(self.reshape_answer_board(answer_board8))
                                        num = 0
                                        # xTrain,yTrainの中身を代入する処理をここに書く。
                                        batch_count_num += 1
                                    # 注意　moveはinputを作成した後にすること。

                                    go_state_obj = rule.move(go_state_obj, current_player, pos_tuple)
                                    rule.move(go_state_obj, current_player, pos_tuple)
                                    # print rule.print_board(go_state_obj)

                                    if (batch_count_sum_all + 98) % 20 == 0 and batch_count_num > 50:
                                        # print xTrain
                                        print "tensorboard writing!"
                                        # train_step.run(feed_dict={x_input: xTrain, y_: yTrain,keep_prob: 0.5})
                                        summary_str, accuracy_value, cross_entropy_value = sess.run([merged, accuracy, cross_entropy],feed_dict={x_input: xTrain, y_: yTrain})
                                        train_writer.add_summary(summary_str, step)
                                        train_writer.flush()
                                        # train_writer.add_summary(summary_str[0],step)
                                        print accuracy_value
                                        print str(float(cross_entropy_value))
                                        # print cross_entropy
                                        # train_step.run(feed_dict={x_input: xTrain, y_: yTrain,keep_prob: 0.5})

                                        print summary_str[0]
                                        batch_count_sum_all += 1
                                        batch_count_num = 0
                                        train_count_num += 1
                                        xTrain = []
                                        yTrain = []

                                        print train_count_num
                                    elif batch_count_num > 50:#TODO modify
                                        #print "batch"
                                        train_step.run(feed_dict={x_input: xTrain, y_: yTrain})
                                        # train_accuracy = accuracy.eval(feed_dict={x_input:xTrain, y_: yTrain, keep_prob: 1.0})
                                        batch_count_sum_all += 1
                                        batch_count_num = 0
                                        train_count_num += 1
                                        xTrain = []
                                        yTrain = []
                                        print train_count_num
                                    if train_count_num > 1500:
                                        train_count_num = 0
                                        ckpt_num += 1
                                        print "SAVED!"
                                        saver.save(sess, './Network_Backup/fast_policy' + str(ckpt_num))

                        except:
                            # traceback.print_exc()
                            f.close()
                            pass
