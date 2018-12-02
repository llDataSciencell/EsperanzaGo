# coding:utf-8
# TODO: batchでまんべんなくデータが含まれるようにする。
# sgf読み込みで新しい棋譜になったら盤面を初期化する
import tensorflow as tf
import pyximport
#TODO: sessは１回しか使えないので、実行するスクリプトを一つにする。
#pyximport.install()
from game import Player
from search import MontecarloSearch, DeepLearningSearch
from go import Go
from input_plane import MakeInputPlane

from forward_prop_network import ForwardPropNetwork
from go import GoVariable
from go import GoStateObject
from numpy import *
import sgf
import random as random
import copy
import os
import traceback

'''パスはどうする？forwardした結果一番良い答えがパスかもしれない'''
import sys
from guppy import hpy
h = hpy()

reload(sys)
sys.stderr.write(str(sys.setdefaultencoding('utf-8')))


class ValueNetworkWithLabel(GoVariable):
    character_list = [chr(i) for i in range(97, 97 + 26)]

    def __init__(self, sess):
        self.rule = Go()
        self.make_input = MakeInputPlane()
        self.batch_size = 70
        #self.policy_network()
        self.value_network(sess)

        self.value_network_train(sess)

    def reshape_board(self, board_array):
        reshaped_boards = []
        for i in xrange(len(board_array)):
            reshaped_boards.append(reshape(board_array[i], 361))
        return reshaped_boards
    def reshape_answer(self, answer):
        return array([answer], dtype=float)
        #return reshape(answer,2)
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
        #90度回転させるために、配列を２次元にした方が良い。input shapeもNone,1,361にする。
        rot90(board_array)
        return board_array

    def get_particular_variables(self, name):
        return {v.name: v for v in tf.all_variables() if v.name.find(name) >= 0}

    def value_network(self, sess):
        print "starting tf.device(/gpu:1)"

        # with tf.device("/gpu:1"):
        with tf.device("/gpu:0"):
            n_out=100
            def weight_variable(shape, variable_name=None):
                """適度にノイズを含んだ（対称性の除去と勾配ゼロ防止のため）重み行列作成関数
                """

                initial = tf.truncated_normal(shape, stddev=0.1)
                return tf.Variable(initial, name=variable_name)

            def bias_variable(shape, variable_name=None):
                """バイアス行列作成関数
                """
                initial = tf.constant(0.1, shape=shape)
                return tf.Variable(initial, name=variable_name)

            def conv2d(x, W, variable_name=None):
                """2次元畳み込み関数
                """
                return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=variable_name)

            def batch_normalization(shape, input):
                eps = 1e-5
                gamma = weight_variable([shape])
                beta = weight_variable([shape])
                mean, variance = tf.nn.moments(input, [0])
                return gamma * (input - mean) / tf.sqrt(variance + eps) + beta

            with tf.variable_scope('value_network'):
                #5290000=23*23*100*100  88200=21*21*
                # データ用可変2階テンソルを用意
                self.x_input_value = tf.placeholder("float", shape=[None, 8, 361])
                # 正解用可変2階テンソルを用意
                self.y_value = tf.placeholder("float", shape=[None, 1])

                # 画像を2次元配列にリシェイプ 第1引数は画像数(-1は元サイズを保存するように自動計算)、縦x横、チャネル
                self.x_image_value = tf.reshape(self.x_input_value, [-1, 19, 19, 8])

                # 畳み込み層のフィルタ重み、引数はパッチサイズ縦、パッチサイズ横、入力チャネル数、出力チャネル数
                # 5x5フィルタで100チャネルを出力（入力は白黒画像なので1チャンネル）
                self.W_conv1_value = weight_variable([5, 5, 8, n_out])  #[5,5,6,50]
                self.b_conv1_value = bias_variable([n_out])  #[50]
                self.h_conv1_value = tf.nn.tanh(conv2d(self.x_image_value, self.W_conv1_value) + self.b_conv1_value)
                self.conv1_bn = batch_normalization(n_out, self.h_conv1_value)

                self.W_conv2_value = weight_variable([5, 5, n_out, n_out])  #[3,3,50,n_out]
                self.b_conv2_value = bias_variable([n_out])
                self.h_conv2_value = tf.nn.tanh(conv2d(self.h_conv1_value, self.W_conv2_value) + self.b_conv2_value)
                self.conv2_bn = batch_normalization(n_out, self.h_conv2_value)

                self.W_conv3_value = weight_variable([3, 3, n_out, n_out])
                self.b_conv3_value = bias_variable([n_out])
                self.h_conv3_value = tf.nn.tanh(conv2d(self.h_conv2_value, self.W_conv3_value) + self.b_conv3_value)
                self.conv3_bn = batch_normalization(n_out, self.h_conv3_value)

                self.W_conv4_value = weight_variable([3, 3, n_out, n_out])
                self.b_conv4_value = bias_variable([n_out])
                self.h_conv4_value = tf.nn.tanh(conv2d(self.h_conv3_value, self.W_conv4_value) + self.b_conv4_value)
                self.conv4_bn = batch_normalization(n_out, self.h_conv4_value)

                self.W_conv5_value = weight_variable([3, 3, n_out, n_out])
                self.b_conv5_value = bias_variable([n_out])
                self.h_conv5_value = tf.nn.tanh(conv2d(self.h_conv4_value, self.W_conv5_value) + self.b_conv5_value)
                self.conv5_bn = batch_normalization(n_out, self.h_conv5_value)

                self.W_conv6_value = weight_variable([3, 3, n_out, n_out])
                self.b_conv6_value = bias_variable([n_out])
                self.h_conv6_value = tf.nn.tanh(conv2d(self.h_conv5_value, self.W_conv6_value) + self.b_conv6_value)
                self.conv6_bn = batch_normalization(n_out, self.h_conv6_value)

                self.W_conv7_value = weight_variable([3, 3, n_out, n_out])
                self.b_conv7_value = bias_variable([n_out])
                self.h_conv7_value = tf.nn.tanh(conv2d(self.h_conv6_value, self.W_conv7_value) + self.b_conv7_value)
                self.conv7_bn = batch_normalization(n_out, self.h_conv7_value)

                self.W_conv8_value = weight_variable([3, 3, n_out, n_out])
                self.b_conv8_value = bias_variable([n_out])
                self.h_conv8_value = tf.nn.tanh(conv2d(self.h_conv7_value, self.W_conv8_value) + self.b_conv8_value)
                self.conv8_bn = batch_normalization(n_out, self.h_conv8_value)

                self.W_conv9_value = weight_variable([3, 3, n_out, n_out])
                self.b_conv9_value = bias_variable([n_out])
                self.h_conv9_value = tf.nn.tanh(conv2d(self.h_conv8_value, self.W_conv9_value) + self.b_conv9_value)
                self.conv9_bn = batch_normalization(n_out, self.h_conv9_value)

                self.W_conv10_value = weight_variable([3, 3, n_out, n_out])
                self.b_conv10_value = bias_variable([n_out])
                self.h_conv10_value = tf.nn.tanh(conv2d(self.h_conv9_value, self.W_conv10_value) + self.b_conv10_value)
                self.conv10_bn = batch_normalization(n_out, self.h_conv10_value)

                self.W_conv11_value = weight_variable([3, 3, n_out, n_out])
                self.b_conv11_value = bias_variable([n_out])
                self.h_conv11_value = tf.nn.tanh(conv2d(self.h_conv10_value, self.W_conv11_value) + self.b_conv11_value)
                self.conv11_bn = batch_normalization(n_out, self.h_conv11_value)

                self.W_conv12_value = weight_variable([3, 3, n_out, n_out])
                self.b_conv12_value = bias_variable([n_out])
                self.h_conv12_value = tf.nn.tanh(conv2d(self.h_conv11_value, self.W_conv12_value) + self.b_conv12_value)
                self.conv12_bn = batch_normalization(n_out, self.h_conv12_value)

                self.W_conv13_value = weight_variable([3, 3, n_out, n_out])
                self.b_conv13_value = bias_variable([n_out])
                self.h_conv13_value = tf.nn.tanh(conv2d(self.h_conv12_value, self.W_conv13_value) + self.b_conv13_value)
                self.conv13_bn = batch_normalization(n_out, self.h_conv13_value)

                self.bn_flat_value = tf.reshape(self.conv13_bn, [-1, 19 * 19 * n_out])  #29*29*n_out



        with tf.device("/gpu:1"):
            def weight_variable(shape, variable_name=None):
                """適度にノイズを含んだ（対称性の除去と勾配ゼロ防止のため）重み行列作成関数
                """

                initial = tf.truncated_normal(shape, stddev=0.1)
                return tf.Variable(initial, name=variable_name)
            def bias_variable(shape, variable_name=None):
                """バイアス行列作成関数
                """
                initial = tf.constant(0.1, shape=shape)
                return tf.Variable(initial, name=variable_name)

            with tf.variable_scope('value_network'):
                # ドロップアウトを指定
                self.keep_prob_value = tf.placeholder("float")
                self.h_fc1_drop_value = tf.nn.dropout(self.bn_flat_value, self.keep_prob_value)

                self.weight_fully_connected2_value = weight_variable([19*19*n_out, 1])
                self.bias_fc2_value = bias_variable([1])
                self.y_conv_value = tf.nn.tanh(tf.matmul(self.h_fc1_drop_value, self.weight_fully_connected2_value) + self.bias_fc2_value)

        #self.logistic_value = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.y_conv_value, self.y_value))))
        #self.cross_entropy_value = -tf.reduce_sum(self.y_value * tf.log(tf.clip_by_value(self.y_conv_value,1e-10,1.0)))
        self.cross_entropy_value = tf.reduce_mean(tf.square(self.y_conv_value - self.y_value))
        self.train_step_value = tf.train.AdamOptimizer(1e-6).minimize(self.cross_entropy_value)
        self.correct_prediction_value = tf.equal(tf.greater(self.y_conv_value, 0), tf.greater(self.y_value, 0))
        #self.correct_prediction_value = tf.equal(tf.argmax(self.y_conv_value, 1), tf.argmax(self.y_value, 1))
        self.accuracy_value = tf.reduce_mean(tf.cast(self.correct_prediction_value, "float"))
        sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver(self.get_particular_variables('value_network'))
        # 指定したパス内の全てのファイルとディレクトリを要素とするリストを返す
        #自分の環境でのsgf_filesへのパスを書く
        tf.summary.scalar("logistic_value", self.cross_entropy_value)
        tf.summary.scalar("accuracy_value", self.accuracy_value)
        self.merged_value = tf.summary.merge_all()
    def value_network_train(self, sess):
        #TODO RL policy同士を戦わせた時の勝率を予測するように、学習（但し、高速policyのsoftmaxでどれくらいの確率で選ばれるかを加味して考える　確率×勝率）
        #tf.summary.scalar("accuracy",accuracy)
        #tf.summary.scalar("correct_prediction",correct_prediction)
        #tf.summary.scalar("train_step",train_step)
        #merged = tf.summary.merge_all()
        players = [Player(0.0, 'human'), Player(1.0, 'human')]
        players[0].next_player = players[1]
        players[1].next_player = players[0]
        #player = players[0]
        rule = Go()
        files = os.listdir(os.getcwd() + "/kifu")

        init = tf.initialize_all_variables()
        xTrain = []
        yTrain = []

        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter('tensorflow_logs/value_train', sess.graph)
            test_writer = tf.summary.FileWriter('tensorflow_logs/value_test', sess.graph)

            sess.run(init)  #If it is first time of learning

            num = 0

            batch_count_num = 0
            train_count_num = 0

            self.saver.restore(sess, './Network_Backup/value_batchnorm')

            make_input = MakeInputPlane()

            step = 0
            ckpt_num = 0
            batch_count_sum_all = 0
            white_count=0
            black_count=0

            for _ in xrange(100):
                continue_kifu_num = 0

                for file_name in files:
                    #print h.heap()

                    continue_kifu_num += 1
                    if continue_kifu_num < 200000:
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
                            go_state_obj = GoStateObject()
                            datastr = f.read()
                            collection = sgf.parse(datastr)
                            flag = False
                            #print "通過"
                            for game in collection:
                                node_length = len(game.nodes)
                                '''
                                if node_length-1 > 230:
                                    interrupt_num = random.randint(230,node_length-2)
                                else:

                                    interrupt_num = random.randint(0, node_length - 2)
                                '''
                                interrupt_num = random.randint(0, node_length - 2)
                                for node in game:
                                    if flag == False:
                                        flag = True
                                        continue
                                    lists = node.properties.values()
                                    #print lists
                                    internal_lists = lists[0]
                                    position = internal_lists[0]
                                    xpos = self.character_list.index(position[0])
                                    ypos = self.character_list.index(position[1])
                                    pos_tuple = (xpos, ypos)
                                    #print xpos,ypos
                                    if node.properties.has_key('B') == True:
                                        current_player = players[0]
                                    elif node.properties.has_key('W') == True:
                                        current_player = players[1]

                                    num += 1

                                    if (batch_count_sum_all + 116) % 20 == 0 and batch_count_num >= self.batch_size:
                                        #print h.heap()
                                        #print xTrain
                                        print "tensorboard writing!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                                        #train_step.run(feed_dict={x_input: xTrain, y_: yTrain,keep_prob: 0.5})
                                        y_label=[]
                                        x_label=[]
                                        print "white:"+str(white_count)+"black:"+str(black_count)

                                        if black > white:
                                            bat_num=white_count*8
                                        elif black < white:
                                            bat_num=black_count*8
                                        else:
                                            bat_num=black_count*8

                                        black=0
                                        white=0
                                        for num in xrange(0,len(yTrain)):
                                            if yTrain[num] == float(1):
                                                black+=1
                                                if black < bat_num:
                                                    y_label.append(self.reshape_answer(yTrain[num]))
                                                    x_label.append(xTrain[num])
                                            elif yTrain[num] == float(-1):
                                                white+=1
                                                if white < bat_num:
                                                    y_label.append(self.reshape_answer(yTrain[num]))
                                                    x_label.append(xTrain[num])
                                                white += 1
                                        print "bat_num"+str(bat_num)
                                        print "y_label"+str(len(y_label))+":x_label"+str(len(x_label))

                                        '''
                                        print "white:"+str(white_count)+"black:"+str(black_count)
                                        for num in xrange(0,len(yTrain)):
                                            y_label.append(self.reshape_answer(float(yTrain[num])*float(white_count)/float(white_count+black_count)) if yTrain[num]==float(1) else self.reshape_answer(float(yTrain[num])*float(black_count)/float(black_count+white_count)))
                                        white_count=0
                                        black_count=0
                                        '''
                                        summary_str, accuracy_val, entropy_val = sess.run([self.merged_value, self.accuracy_value, self.cross_entropy_value],feed_dict={self.x_input_value: x_label, self.y_value: y_label, self.keep_prob_value: 1.0})

                                        train_writer.add_summary(summary_str, step)
                                        train_writer.flush()
                                        #train_writer.add_summary(summary_str[0],step)
                                        print "accuracy:"+str(accuracy_val)
                                        print "予測："+str(float(entropy_val))
                                        print sess.run(self.y_conv_value,feed_dict={self.x_input_value: xTrain,self.keep_prob_value: 1.0})

                                        batch_count_sum_all += 1
                                        batch_count_num = 0
                                        train_count_num += 1
                                        xTrain = []
                                        yTrain = []

                                        print train_count_num
                                    elif batch_count_num >= self.batch_size:
                                        y_label=[]
                                        x_label=[]
                                        print "white:"+str(white_count)+"black:"+str(black_count)
                                        black=0
                                        white=0
                                        diff=black_count-white_count

                                        if black > white:
                                            bat_num=white_count*8
                                        elif black < white:
                                            bat_num=black_count*8
                                        else:
                                            bat_num=black_count*8
                                        #30 * 8 = 240 batch size if black_count = 13,white_count = 17 then total num is (13+13)*8=208 batch size.
                                        for num in xrange(0,len(yTrain)):
                                            if yTrain[num] == float(1):
                                                black+=1
                                                if black <= bat_num:
                                                    y_label.append(self.reshape_answer(yTrain[num]))
                                                    x_label.append(xTrain[num])
                                            elif yTrain[num] == float(-1):
                                                white+=1
                                                if white <= bat_num:
                                                    y_label.append(self.reshape_answer(yTrain[num]))
                                                    x_label.append(xTrain[num])
                                            #y_label.append(self.reshape_answer(float(yTrain[num])*float(white_count)/float(white_count+black_count)) if yTrain[num]==float(1) else self.reshape_answer(float(yTrain[num])*float(black_count)/float(black_count+white_count)))
                                        print "y_label"+str(len(y_label))+":x_label"+str(len(x_label))

                                        white_count=0
                                        black_count=0

                                        self.train_step_value.run(feed_dict={self.x_input_value: x_label, self.y_value: y_label,self.keep_prob_value: 0.5})
                                        y_label=[]
                                        #train_accuracy = accuracy.eval(feed_dict={x_input:xTrain, y_: yTrain, keep_prob: 1.0})
                                        batch_count_sum_all += 1
                                        batch_count_num = 0
                                        train_count_num += 1
                                        xTrain = []
                                        yTrain = []
                                        print train_count_num

                                    if num >= interrupt_num and current_player.player_id==float(0.0):#length error might be occured
                                        #print(datastr.find('RE['))
                                        #print "data.find('RE')"
                                        #print "game ends" + str(num)
                                        #print "current player at mid is BLACK"
                                        answer=0
                                        if 'RE[B' in datastr:
                                            answer = float(1)
                                        elif 'RE[黑' in datastr:
                                            answer=float(1)
                                        elif 'RE[黒' in datastr:
                                            answer=float(1)
                                        elif '黒中盤勝' in datastr:
                                            answer = float(1)
                                        elif '黑中盘胜' in datastr:
                                            answer = float(1)
                                        elif '白中盤勝' in datastr:
                                            answer=float(-1)
                                        elif '白中盘胜' in datastr:
                                            answer=float(-1)
                                        elif 'RE[白' in datastr:
                                            answer = float(-1)
                                        elif 'RE[W' in datastr:
                                            answer=float(-1)
                                        elif 'RE[白' in datastr:
                                            answer=float(-1)
                                        else:
                                            #print datastr
                                            answer = float(-0.00001)

                                        if answer==float(1):
                                            black_count+=1
                                        elif answer==float(-1):
                                            white_count+=1

                                        #current_player使わない！バグの原因になる 黒から見た場合に固定
                                        input_board = make_input.generate_input(go_state_obj, players[0])
                                        xTrain.append(self.reshape_board(input_board))
                                        yTrain.append(answer)

                                        input_board2 = self.rotate90_input(input_board)
                                        xTrain.append(self.reshape_board(input_board2))
                                        yTrain.append(answer)

                                        input_board3 = self.rotate90_input(input_board2)
                                        xTrain.append(self.reshape_board(input_board3))
                                        yTrain.append(answer)

                                        input_board4 = self.rotate90_input(input_board3)
                                        xTrain.append(self.reshape_board(input_board4))
                                        yTrain.append(answer)

                                        input_board5 = self.invert_board_input(input_board4)
                                        xTrain.append(self.reshape_board(input_board5))
                                        yTrain.append(answer)

                                        input_board6 = self.rotate90_input(input_board5)
                                        xTrain.append(self.reshape_board(input_board6))
                                        yTrain.append(answer)

                                        input_board7 = self.rotate90_input(input_board6)
                                        xTrain.append(self.reshape_board(input_board7))
                                        yTrain.append(answer)

                                        input_board8 = self.rotate90_input(input_board7)
                                        xTrain.append(self.reshape_board(input_board8))
                                        yTrain.append(answer)

                                        num = 0
                                        #xTrain,yTrainの中身を代入する処理をここに書く。
                                        batch_count_num += 1

                                        raise Exception


                                    go_state_obj = rule.move(go_state_obj, current_player, pos_tuple)
                                    rule.move(go_state_obj, current_player, pos_tuple)


                                    if train_count_num > 1000:
                                        train_count_num = 0
                                        ckpt_num += 1
                                        print "SAVED!!!!!!!!!!!!!!!!!!!!!!!!!!"
                                        self.saver.save(sess,'./Network_Backup/value_batchnorm' + str(ckpt_num))

                        except:
                            #traceback.print_exc()
                            f.close()



def _main(self, sess):
    #self.saver.restore(sess, os.getcwd() + "/model.ckpt")  # モデルの読み込み　作業の再開時にコメントアウト
    #self.saver_value.restore(sess, os.getcwd() + "/model_value.ckpt")  # モデルの読み込み　作業の再開時にコメントアウト

    print "gtp_io"
    #self._test(sess)
    self.value_network_train(sess)
