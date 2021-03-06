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
from tensorflow.python import control_flow_ops
# sysモジュールをリロードする
reload(sys)
# デフォルトの文字コードを変更する
sys.stderr.write(str(sys.setdefaultencoding('utf-8')))


# デフォルトの文字コードを出力する
# from guppy import hpy
# h = hpy()


class Train(GoVariable):
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

    def weight_variable(self, shape):
        """適度にノイズを含んだ（対称性の除去と勾配ゼロ防止のため）重み行列作成関数
        """

        initial = tf.truncated_normal(shape, stddev=0.05)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """バイアス行列作成関数
        """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        """2次元畳み込み関数
        """
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def batch_norm(x, n_out, phase_train, scope='bn'):
        """
        Batch normalization on convolutional maps.
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            phase_train: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        with tf.variable_scope(scope):
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                               name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

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

        with tf.device("/gpu:0"):
            phase_train = tf.placeholder(tf.bool, name='phase_train')
            n_out=96
            with tf.variable_scope('policy_auto'):
                # 5290000=23*23*100*100  88200=21*21*
                # データ用可変2階テンソルを用意
                x_input = tf.placeholder("float", shape=[None, 8, 361])
                # 正解用可変2階テンソルを用意
                y_ = tf.placeholder("float", shape=[None, 361])

                # 画像を2次元配列にリシェイプ 第1引数は画像数(-1は元サイズを保存するように自動計算)、縦x横、チャネル
                x_image = tf.reshape(x_input, [-1, 19, 19, 8])
                x_image_pad = tf.pad(x_image, [[0, 0], [2, 2], [2, 2], [0, 0]])
                # 畳み込み層のフィルタ重み、引数はパッチサイズ縦、パッチサイズ横、入力チャネル数、出力チャネル数
                # 5x5フィルタで100チャネルを出力（入力は白黒画像なので1チャンネル）
                W_conv1 = self.weight_variable([5, 5, 8, 96])  # [5,5,6,50]
                b_conv1 = self.bias_variable([96])  # [50]
                h_conv1 = tf.nn.relu(self.conv2d(x_image_pad, W_conv1) + b_conv1)
                conv1_bn = self.batch_norm(h_conv1, n_out, phase_train)
                #h_pool1 = self.max_pool_2x2(h_conv1)
                # 次の層の畳み込み処理を行う前に、paddingを実行。
                # 3層目 畳み込み層

                # パッチサイズ縦、パッチサイズ横、入力チャネル（枚数）、出力チャネル（出力の枚数）
                # 3x3フィルタで64チャネルを出力
                W_conv2 = self.weight_variable([3, 3, 96, 96])  # [3,3,50,100]
                b_conv2 = self.bias_variable([96])
                h_conv2 = tf.nn.relu(self.conv2d(conv1_bn, W_conv2) + b_conv2)
                conv2_bn = self.batch_norm(h_conv2, n_out, phase_train)
                #h_pool2 = self.max_pool_2x2(h_conv2)
                # 次の層の畳み込み処理を行う前に、paddingを実行。

                ###5層目 畳み込み層
                W_conv3 = self.weight_variable([3, 3, 96, 96])
                b_conv3 = self.bias_variable([96])
                h_conv3 = tf.nn.relu(self.conv2d(conv2_bn, W_conv3) + b_conv3)
                conv3_bn = self.batch_norm(h_conv3, n_out, phase_train)
                #h_pool3 = self.max_pool_2x2(h_conv3)

                ###7層目　畳み込み層
                W_conv4 = self.weight_variable([3, 3, 96, 96])
                b_conv4 = self.bias_variable([96])
                h_conv4 = tf.nn.relu(self.conv2d(conv3_bn, W_conv4) + b_conv4)

                conv4_bn = self.batch_norm(h_conv4, n_out, phase_train)
                #h_pool4 = self.max_pool_2x2(h_conv4)

                #h_pool4_flat = tf.reshape(h_pool4, [-1, 23 * 23 * 96])  # 29*29*100
                h_pool4_flat = tf.reshape(conv4_bn, [-1, 23 * 23 * 96])
                weight_fully_connected1 = self.weight_variable([23 * 23 * 96, 1000])
                bias_fc1 = self.weight_variable([1000])
                hidden_fully_connect1 = tf.nn.relu(tf.matmul(h_pool4_flat, weight_fully_connected1) + bias_fc1)

                keep_prob = tf.placeholder("float")
                h_fc1_drop = tf.nn.dropout(hidden_fully_connect1, keep_prob)

                weight_fully_connected2 = self.weight_variable([1000, 361])
                bias_fc2 = self.bias_variable([361])
                y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, weight_fully_connected2) + bias_fc2)
        cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
        # cross_entropy=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,labels=y_))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(self.get_particular_variables('policy_auto'))
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

            saver.restore(sess, './Network_Backup/policy_legal_move')

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
                    if continue_kifu_num < 150:
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
                                        summary_str, accuracy_value, cross_entropy_value = sess.run(
                                            [merged, accuracy, cross_entropy],
                                            feed_dict={x_input: xTrain, y_: yTrain, keep_prob: 0.5})
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
                                    elif batch_count_num > 50:
                                        train_step.run(feed_dict={x_input: xTrain, y_: yTrain, keep_prob: 0.5})
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
                                        saver.save(sess, './Network_Backup/policy_legal_move' + str(ckpt_num))

                        except:
                            # traceback.print_exc()
                            f.close()
                            pass
