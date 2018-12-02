# coding:utf-8
# TODO: batchでまんべんなくデータが含まれるようにする。
# sgf読み込みで新しい棋譜になったら盤面を初期化する
import pyximport

pyximport.install()
import sgf  # Please install "sgf 0.5". You can install it by using the command of "pip install sgf".
import sys
import re
import os  # osモジュールのインポート
from game import Player
from game import State
from search import MontecarloSearch, DeepLearningSearch
from go import Go
from input_plane import MakeInputPlane
import tensorflow as tf
import math
from go import GoVariable
from go import GoStateObject
from numpy import *

'''パスはどうする？forwardした結果一番良い答えがパスかもしれない'''
import sys

reload(sys)
sys.stderr.write(str(sys.setdefaultencoding('utf-8')))

class ForwardPropNetwork(GoVariable):
    character_list = [chr(i) for i in range(97, 97 + 26)]

    def __init__(self,sess):
        self.make_input = MakeInputPlane()

        self.train(sess)
        sys.stderr.write(str("train_ended"))
        self._main(sess)

    def reshape_board(self, board_array):
        reshaped_boards = []
        for i in xrange(len(board_array)):
            reshaped_boards.append(reshape(board_array[i], 361))

        return reshaped_boards

    def weight_variable(self, shape):
        """適度にノイズを含んだ（対称性の除去と勾配ゼロ防止のため）重み行列作成関数
        """

        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """バイアス行列作成関数
        """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        """2次元畳み込み関数
        """
        return tf.nn.conv2d(x,
                            W,
                            strides=[1, 1, 1, 1],  # 真ん中2つが縦横のストライド 1,1で画像が小さくならない
                            padding='SAME')

    def max_pool_2x2(self, x):
        """2x2マックスプーリング関数
        """

        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

    def get_particular_variables(self,name):
        return {v.name: v for v in tf.global_variables() if v.name.find(name) >= 0}

    def batch_normalization(self,shape, input):
        eps = 1e-5
        gamma = self.weight_variable([shape])
        beta =  self.weight_variable([shape])
        mean, variance = tf.nn.moments(input, [0])
        return gamma * (input - mean) / tf.sqrt(variance + eps) + beta

    def train(self,sess):
        with tf.device("/cpu:0"):
             with tf.variable_scope('fast'):
                phase_train = tf.placeholder(tf.bool, name='phase_train')
                # 5290000=23*23*100*100  88200=21*21*
                # データ用可変2階テンソルを用意
                self.x_input = tf.placeholder("float", shape=[None, 8, 361])
                # 正解用可変2階テンソルを用意
                self.y_ = tf.placeholder("float", shape=[None, 361])

                # 画像を2次元配列にリシェイプ 第1引数は画像数(-1は元サイズを保存するように自動計算)、縦x横、チャネル
                #x_image = tf.reshape(x_input, [-1, 19, 19, 8])
                self.x_input_flat = tf.reshape(self.x_input, [-1, 19 * 19 * 8])
                self.weight_fully_connected1 = self.weight_variable([19 * 19 * 8,19*19+100])
                self.bias_fc1 = self.weight_variable([19*19+100])
                self.hidden_fully_connect1 = tf.nn.relu(tf.matmul(self.x_input_flat, self.weight_fully_connected1) + self.bias_fc1)
                #bn_hidden1=batch_normalization(19*19+100,hidden_fully_connect1)
                self.weight_fully_connected2 = self.weight_variable([19*19+100, 361])
                self.bias_fc2 = self.bias_variable([361])
                self.y_conv = tf.nn.softmax(tf.matmul(self.hidden_fully_connect1, self.weight_fully_connected2) + self.bias_fc2)
        self.cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y_conv))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        self.init = tf.global_variables_initializer()
        #self.saver = tf.train.Saver({'x_input':self.x_input,'y_': self.y_,'x_image':self.x_image,'x_image_pad':self.x_image_pad,'W_conv1':self.W_conv1,'b_conv1':self.b_conv1,'h_conv1':self.h_conv1,'h_pool1':self.h_pool1,'W_conv2':self.W_conv2,'b_conv2':self.b_conv2,'h_conv2':self.h_conv2,'h_pool2':self.h_pool2,'W_conv3':self.W_conv3,'b_conv3':self.b_conv3,'h_conv3':self.h_conv3,'h_pool3':self.h_pool3,'W_conv4':self.W_conv4,'b_conv4':self.b_conv4,'h_conv4':self.h_conv4,'h_pool4':self.h_pool4,'weight_fully_connected1':self.weight_fully_connected1,'bias_fc1':self.bias_fc1,'h_pool4_flat':self.h_pool4_flat,'hidden_fully_connect1':self.hidden_fully_connect1,'weight_fully_connected2':self.weight_fully_connected2,'bias_fc2':self.bias_fc2,'y_conv':self.y_conv,'train_step':self.train_step,'correct_prediction':self.correct_prediction ,'accuracy':self.accuracy,'init':self.init})
        self.saver = tf.train.Saver(self.get_particular_variables('fast'))

    def search_deep_learning(self,sess, go_state_obj, current_player):
        xTrain = [self.reshape_board(self.make_input.generate_input(go_state_obj, current_player))]

        output_array = sess.run(self.y_conv, feed_dict={self.x_input: xTrain})
        #two_dimensional_array = tf.reshape(output_array, [-1, 19, 19, 1])
        return output_array

    letter_coords = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']

    def coord_to_str(self, row, col):
        return self.letter_coords[row] + str(col + 1)  # 数字をcharacterに変換
    def gtp_io(self, sess):
        players = [Player(0, 'human'), Player(1, 'human')]
        players[0].next_player = players[1]
        players[1].next_player = players[0]
        player = players[0]
        state = Go()
        go_state_obj = GoStateObject()
        # search_algorithm = SimpleSearch()
        search_algorithm = DeepLearningSearch()
        """ Main loop which communicates to gogui via GTP"""
        known_commands = ['boardsize', 'clear_board', 'komi', 'play', 'genmove',
                          'final_score', 'quit', 'name', 'version', 'known_command',
                          'list_commands', 'protocol_version', 'gogui-analyze_commands']
        analyze_commands = ["gfx/Predict Final Ownership/predict_ownership",
                            "none/Load New SGF/loadsgf"]
        # print("starting main.py: loading %s" %sgf_file,file=sys.stderr)
        # output_file = open("output.txt", "wb")
        # output_file.write("intializing\n")
        sys.stderr.write(str("19路盤専用！"))

        while True:
            try:
                line = raw_input().strip()
                # print line
                # output_file.write(line + "\n")
            except EOFError:
                # output_file.write('Breaking!!\n')
                break
            if line == '':
                continue
            command = [s.lower() for s in line.split()]
            # print command
            if re.match('\d+', command[0]):
                command = command[1:]

            ret = ''
            if command[0] == "clear_board":
                state = Go()
                go_state_obj = GoStateObject()
                ret = "= \n\n"
            elif command[0] == "komi":
                ret = "= \n\n"
            elif command[0] == "play":  # playにバグあり "play w C19"が渡された時はx座標,yの順番
                if command[2] == "pass":
                    sys.stderr.write(str(state.print_board(go_state_obj)))
                else:
                    if command[1][0] == "b":
                        player = players[0]
                    elif command[1][0] == "w":
                        player = players[1]

                    x = self.letter_coords.index(command[2][0].upper())  # リストは0からスタート
                    y = 18 - (int(command[2][1:]) - 1)

                    go_state_obj = state.move(go_state_obj, player, (x, y))
                    sys.stderr.write(str(player.player_id))
                    # print go_state_obj._board
                    # print state.print_board(go_state_obj)

                ret = "= \n\n"

            elif command[0] == "genmove":
                if command[1] == "b":
                    player = players[0]
                elif command[1] == "w":
                    player = players[1]
                (go_state_obj, move) = search_algorithm.next_move(self,sess, go_state_obj, player)
                # tup=[move[0],move[1]]
                if move == "pass":
                    ret = "pass"
                else:
                    ret = self.coord_to_str(move[0], 18 - move[1])
                    sys.stderr.write(str("move x:" + str(move[0]) + "\n"))
                    sys.stderr.write(str("move: y" + str(move[1]) + "\n"))

                    sys.stderr.write(str(state.print_board(go_state_obj)))

                ret = '= ' + ret + '\n\n'

            elif command[0] == "final_score":
                # print("final_score not implemented", file=sys.stderr)
                ret = "= \n\n"
            elif command[0] == "name":
                ret = '= EsperanzaGo\n\n'
            elif command[0] == "predict_ownership":
                # ownership_prediction = driver.evaluate_current_board()
                # ret = influence_str(ownership_prediction)
                ret = "= \n\n"
            elif command[0] == "version":
                ret = '= 1.0\n\n'
            elif command[0] == "list_commands":
                # ret = '= \n'.join(known_commands)
                ret = "= boardsize\nclear_board\nquit\nprotocol_version\n" + "name\nversion\nlist_commands\nkomi\ngenmove\nplay\n\n"
            elif command[0] == "gogui-analyze_commands":
                ret = '\n'.join(analyze_commands)

            elif command[0] == "known_command":
                ret = 'true' if command[1] in known_commands else 'false'
            elif command[0] == "protocol_version":
                ret = '= 2\n\n'
            elif command[0] == "boardsize":
                ret = '= \n\n'
            elif command[0] == "quit":
                # ret = '= \n\n'
                # print ret
                # print('=%s \n\n' % (cmdid,), end='')
                exit(0)
                break
            else:
                # print 'Warning: Ignoring unknown command'
                pass
            print ret
            sys.stdout.flush()
    def test(self, sess):
        players = [Player(0.0, 'human'), Player(1.0, 'human')]
        players[0].next_player = players[1]
        players[1].next_player = players[0]
        rule = Go()

        player = players[0]
        go_state_obj = GoStateObject()

        x_train = [[self.reshape_board(self.make_input.generate_input(go_state_obj, player))]]
        #print sess.run(self.y_conv, feed_dict={self.x_input: x_train})

    def _main(self,sess):
        self.saver.restore(sess,'./Network_Backup/fast_policy')
        #self.test(sess)
        #print "gtp_io"
        #gtp_io(sess)は普段はコメントアウトすること。
        #self.gtp_io(sess)


#if __name__ == "__main__":
    #ForwardPropNetwork(sess)
