# coding:utf-8
# TODO: batchでまんべんなくデータが含まれるようにする。
# sgf読み込みで新しい棋譜になったら盤面を初期化する
#import pyximport

#TODO: sessは１回しか使えないので、実行するスクリプトを一つにする。
pyximport.install()
from game import Player
from search import MontecarloSearch, DeepLearningSearch
from go import Go
from input_plane import MakeInputPlane
import tensorflow as tf
from forward_prop_network import ForwardPropNetwork
from go import GoVariable
from go import GoStateObject
from numpy import *
import random as random
import copy

'''パスはどうする？forwardした結果一番良い答えがパスかもしれない'''
import sys

reload(sys)
sys.stderr.write(str(sys.setdefaultencoding('utf-8')))


class ValueNetwork(GoVariable):
    character_list = [chr(i) for i in range(97, 97 + 26)]

    def __init__(self, sess):
        self.rule = Go()
        self.make_input = MakeInputPlane()

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
    def weight_variable(self, shape, variable_name=None):
        """適度にノイズを含んだ（対称性の除去と勾配ゼロ防止のため）重み行列作成関数
        """

        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=variable_name)

    def bias_variable(self, shape, variable_name=None):
        """バイアス行列作成関数
        """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=variable_name)

    def conv2d(self, x, W, variable_name=None):
        """2次元畳み込み関数
        """
        return tf.nn.conv2d(x,
                            W,
                            strides=[1, 1, 1, 1],  # 真ん中2つが縦横のストライド 1,1で画像が小さくならない
                            padding='SAME', name=variable_name)

    def max_pool_2x2(self, x, variable_name=None):
        """2x2マックスプーリング関数
        """

        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name=variable_name)
    def get_particular_variables(self,name):
        return {v.name: v for v in tf.all_variables() if v.name.find(name) >= 0}

    def value_network(self, sess):

        print "starting tf.device(/gpu:1)"

        # with tf.device("/gpu:1"):
        with tf.device("/gpu:1"):
            with tf.variable_scope('value_network'):
                #5290000=23*23*100*100  88200=21*21*
                # データ用可変2階テンソルを用意
                self.x_input_value = tf.placeholder("float", shape=[None, 7, 361])
                # 正解用可変2階テンソルを用意
                self.y_value = tf.placeholder("float", shape=[None, 1])

                # 画像を2次元配列にリシェイプ 第1引数は画像数(-1は元サイズを保存するように自動計算)、縦x横、チャネル
                self.x_image_value = tf.reshape(self.x_input_value, [-1, 19, 19, 7])
                # 畳み込み層のフィルタ重み、引数はパッチサイズ縦、パッチサイズ横、入力チャネル数、出力チャネル数
                # 5x5フィルタで100チャネルを出力（入力は白黒画像なので1チャンネル）
                self.W_conv1_value = self.weight_variable([5, 5, 7, 75])  #[5,5,6,50]
                self.b_conv1_value = self.bias_variable([75])  #[50]
                self.h_conv1_value = tf.nn.relu(self.conv2d(self.x_image_value, self.W_conv1_value) + self.b_conv1_value)
                self.h_pool1_value = self.max_pool_2x2(self.h_conv1_value)

                # パッチサイズ縦、パッチサイズ横、入力チャネル（枚数）、出力チャネル（出力の枚数）
                # 3x3フィルタで75チャネルを出力
                self.W_conv2_value = self.weight_variable([5, 5, 75, 75])  #[3,3,50,100]
                self.b_conv2_value = self.bias_variable([75])
                self.h_conv2_value = tf.nn.relu(self.conv2d(self.h_pool1_value, self.W_conv2_value) + self.b_conv2_value)
                self.h_pool2_value = self.max_pool_2x2(self.h_conv2_value)

                ###5層目 畳み込み層
                self.W_conv3_value = self.weight_variable([3, 3, 75, 75])
                self.b_conv3_value = self.bias_variable([75])
                self.h_conv3_value = tf.nn.relu(self.conv2d(self.h_pool2_value, self.W_conv3_value) + self.b_conv3_value)
                self.h_pool3_value = self.max_pool_2x2(self.h_conv3_value)

                ###7層目　畳み込み層
                self.W_conv4_value = self.weight_variable([3, 3, 75, 75])
                self.b_conv4_value = self.bias_variable([75])
                self.h_conv4_value = tf.nn.relu(self.conv2d(self.h_pool3_value, self.W_conv4_value) + self.b_conv4_value)
                self.h_pool4_value = self.max_pool_2x2(self.h_conv4_value)

                self.W_conv5_value = self.weight_variable([3, 3, 75, 75])
                self.b_conv5_value = self.bias_variable([75])
                self.h_conv5_value = tf.nn.relu(
                    self.conv2d(self.h_pool4_value, self.W_conv5_value) + self.b_conv5_value)
                self.h_pool5_value = self.max_pool_2x2(self.h_conv5_value)

                self.h_pool6_flat_value = tf.reshape(self.h_pool5_value, [-1, 19 * 19 * 75])  #29*29*100
                self.weight_fully_connected1_value = self.weight_variable([19 * 19 * 75, 1000])
                self.bias_fc1_value = self.weight_variable([1000])
                self.hiden_fully_connect1_value = tf.nn.relu(tf.matmul(self.h_pool6_flat_value, self.weight_fully_connected1_value) + self.bias_fc1_value)

                # ドロップアウトを指定
                self.keep_prob_value = tf.placeholder("float")
                self.h_fc1_drop_value = tf.nn.dropout(self.hiden_fully_connect1_value, self.keep_prob_value)

                self.weight_fully_connected2_value = self.weight_variable([1000, 1])
                self.bias_fc2_value = self.bias_variable([1])
                self.y_conv_value = tf.nn.tanh(tf.matmul(self.h_fc1_drop_value,self.weight_fully_connected2_value) + self.bias_fc2_value)

        self.cross_entropy_value = -tf.reduce_sum(self.y_value * tf.log(tf.clip_by_value(self.y_conv_value, 1e-10, 1.0)))
        self.train_step_value = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy_value)
        self.correct_prediction_value = tf.equal(tf.argmax(self.y_conv_value, 1), tf.argmax(self.y_value, 1))
        self.accuracy_value = tf.reduce_mean(tf.cast(self.correct_prediction_value, "float"))
        sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver(self.get_particular_variables('value_network'))
        # 指定したパス内の全てのファイルとディレクトリを要素とするリストを返す
        #自分の環境でのsgf_filesへのパスを書く
        tf.summary.scalar("cross_entropy", self.cross_entropy_value)
        tf.summary.scalar("accuracy", self.accuracy_value)
    def value_network_train(self, sess):
        #TODO RL policy同士を戦わせた時の勝率を予測するように、学習（但し、高速policyのsoftmaxでどれくらいの確率で選ばれるかを加味して考える　確率×勝率）
        players = [Player(0.0, 'human'), Player(1.0, 'human')]
        players[0].next_player = players[1]
        players[1].next_player = players[0]
        #player = players[0]
        rule = Go()

        xTrain = []
        yTrain = []

        self.merged_value = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('tensorflow_logs/train',sess.graph)
        test_writer  = tf.summary.FileWriter('tensorflow_logs/test',sess.graph)

        num = 0
        batch_count_sum_all=0
        batch_count_num = 0
        train_count_num = 0
        step=0

        make_input = MakeInputPlane()

        players = [Player(0, 'human'), Player(1, 'human')]
        players[0].next_player = players[1]
        players[1].next_player = players[0]
        player = players[0]

        search_algorithm = DeepLearningSearch()
        forward_prop_network = ForwardPropNetwork(sess)
        ckpt_num=0
        for _ in xrange(100000):
            go_state_obj = GoStateObject()

            before_move = None

            for i in xrange(random.randint(0, 200)):
                if i % 3 == 0:
                    next_moves = self.rule.next_moves(go_state_obj, player, True)

                    move = random.choice(next_moves)

                    if move == "pass" and before_move == "pass":
                        break

                    go_state_obj = self.rule.move(go_state_obj, player, move)
                    print self.rule.print_board(go_state_obj)
                    player = player.next_player
                    before_move = move
                else:
                    (go_state_obj, move) = search_algorithm.next_move(forward_prop_network, sess, go_state_obj, player)

                    #print self.rule.print_board(go_state_obj)

                    if move == "pass" and before_move == "pass":
                        break

                    go_state_obj = self.rule.move(go_state_obj, player, move)
                    print self.rule.print_board(go_state_obj)
                    player = player.next_player
                    before_move = move
            current_player = copy.deepcopy(player)
            input_board = make_input.generate_input(go_state_obj, current_player)
            print self.rule.print_board(go_state_obj)
            print "player_id:"
            print str(current_player.player_id)
            for playout_num in xrange(200):
                (go_state_obj, move) = search_algorithm.next_move(forward_prop_network, sess, go_state_obj, player)

                #print self.rule.print_board(go_state_obj)

                if move == "pass" and before_move == "pass":
                    break

                go_state_obj = self.rule.move(go_state_obj, player, move)
                print self.rule.print_board(go_state_obj)
                player = player.next_player
                before_move = move

            bw_score = self.rule.count_territory(go_state_obj)
            #If AI player wins,return 1.If Ai lose,return 0.
            if current_player.player_id == self._BLACK:
                score = bw_score[0] - bw_score[1]  #black-white
            elif current_player.player_id == self._WHITE:
                score = bw_score[1] - bw_score[0]

            if score >= 0:
                answer=float(1)
            else:
                answer=float(0)
            print self.rule.print_board(go_state_obj)
            print "answer 1->win     0->lose"
            print answer
            xTrain.append(self.reshape_board(input_board))
            yTrain.append(self.reshape_answer(answer))

            input_board2 = self.rotate90_input(input_board)
            xTrain.append(self.reshape_board(input_board2))
            yTrain.append(self.reshape_answer(answer))

            input_board3 = self.rotate90_input(input_board2)
            xTrain.append(self.reshape_board(input_board3))
            yTrain.append(self.reshape_answer(answer))

            input_board4 = self.rotate90_input(input_board3)
            xTrain.append(self.reshape_board(input_board4))
            yTrain.append(self.reshape_answer(answer))

            input_board5 = self.invert_board_input(input_board4)
            xTrain.append(self.reshape_board(input_board5))
            yTrain.append(self.reshape_answer(answer))

            input_board6 = self.rotate90_input(input_board5)
            xTrain.append(self.reshape_board(input_board6))
            yTrain.append(self.reshape_answer(answer))

            input_board7 = self.rotate90_input(input_board6)
            xTrain.append(self.reshape_board(input_board7))
            yTrain.append(self.reshape_answer(answer))

            input_board8 = self.rotate90_input(input_board7)
            xTrain.append(self.reshape_board(input_board8))
            yTrain.append(self.reshape_answer(answer))

            num = 0
            #xTrain,yTrainの中身を代入する処理をここに書く。
            batch_count_num += 1
            step+=1
            if (batch_count_sum_all + 101) % 30 == 0 and batch_count_num == 30:
                #print xTrain
                print "tensorboard writing!"
                #train_step.run(feed_dict={x_input: xTrain, y_: yTrain,keep_prob: 0.5})

                summary_str, accuracy_value, cross_entropy_value = self.train_step_value.run([self.merged_value, self.accuracy_value, cross_entropy_value],
                                                                            feed_dict={self.x_input: xTrain, self.y_value: yTrain, self.keep_prob: 1.0})
                train_writer.add_summary(summary_str, step)
                train_writer.flush()
                #train_writer.add_summary(summary_str[0],step)
                print accuracy_value
                print str(float(cross_entropy_value))
                #print cross_entropy
                #train_step.run(feed_dict={x_input: xTrain, y_: yTrain,keep_prob: 0.5})

                print summary_str[0]
                batch_count_sum_all += 1
                batch_count_num = 0
                train_count_num += 1
                xTrain = []
                yTrain = []

                print train_count_num
            elif batch_count_num == 30:
                self.train_step_value.run(feed_dict={self.x_input_value: xTrain, self.y_value: yTrain, self.keep_prob_value: 0.5})
                #train_accuracy = accuracy.eval(feed_dict={x_input:xTrain, y_: yTrain, keep_prob: 1.0})
                batch_count_sum_all += 1
                batch_count_num = 0
                train_count_num += 1
                xTrain = []
                yTrain = []
                print train_count_num
            if train_count_num > 2000:
                train_count_num = 0
                ckpt_num += 1
                print "SAVED!"
                self.saver.save(sess, './Network_Backup/value_network_5x5_conv' + str(ckpt_num))




def _main(self, sess):
    #self.saver.restore(sess, os.getcwd() + "/model.ckpt")  # モデルの読み込み　作業の再開時にコメントアウト
    #self.saver_value.restore(sess, os.getcwd() + "/model_value.ckpt")  # モデルの読み込み　作業の再開時にコメントアウト

    print "gtp_io"
    #self._test(sess)
    self.value_network_train(sess)
