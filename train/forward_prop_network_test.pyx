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
import sys
# sysモジュールをリロードする
reload(sys)
# デフォルトの文字コードを変更する
sys.stderr.write(str(sys.setdefaultencoding('utf-8')))
'''パスはどうする？forwardした結果一番良い答えがパスかもしれない'''

# import input_data#delete
class ForwardPropTest(GoVariable):
    character_list = [chr(i) for i in range(97, 97 + 26)]

    def __init__(self):
        '''
        # self.train()
        self.players = [Player(0.0, 'human'), Player(1.0, 'human')]
        self.players[0].next_player = self.players[1]
        self.players[1].next_player = self.players[0]
        # player = players[0]
        self.rule = Go()
        '''
        self.make_input = MakeInputPlane()

        self.train()
        print "train_ended"
        self._main()

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

        return tf.nn.max_pool(x, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

    def train(self):
        # mnistのダウンロード
        # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        print "starting tf.device(/gpu:1)"
        sess = tf.InteractiveSession()
        # with tf.device("/gpu:1"):
        with tf.device("/cpu:0"):
            # 5290000=23*23*100*100  88200=21*21*
            # データ用可変2階テンソルを用意
            self.x_input = tf.placeholder("float", shape=[6, 361])
            # 正解用可変2階テンソルを用意
            self.y_ = tf.placeholder("float", shape=[361])

            # 画像を2次元配列にリシェイプ 第1引数は画像数(-1は元サイズを保存するように自動計算)、縦x横、チャネル
            self.x_image = tf.reshape(self.x_input, [-1, 19, 19, 6])

            print self.x_image

            self.x_image_pad = tf.pad(self.x_image, [[0, 0], [2, 2], [2, 2], [0, 0]])

            ### 1層目 畳み込み層
            ### 1層目 畳み込み層

            # 畳み込み層のフィルタ重み、引数はパッチサイズ縦、パッチサイズ横、入力チャネル数、出力チャネル数
            # 5x5フィルタで100チャネルを出力（入力は白黒画像なので1チャンネル）
            self.W_conv1 = self.weight_variable([5, 5, 6, 50])
            # 畳み込み層のバイアス
            self.b_conv1 = self.bias_variable([50])
            # 活性化関数ReLUでの畳み込み層を構築
            self.h_conv1 = tf.nn.relu(self.conv2d(self.x_image_pad, self.W_conv1) + self.b_conv1)
            # 4次元のpaddingをする
            ### 2層目 プーリング層
            # 2x2のマックスプーリング層を構築
            self.h_pool1 = self.max_pool_2x2(self.h_conv1)
            # 次の層の畳み込み処理を行う前に、paddingを実行。
            tf.pad(self.h_pool1, [[0, 0], [1, 1], [1, 1], [0, 0]])

            ### 3層目 畳み込み層

            # パッチサイズ縦、パッチサイズ横、入力チャネル（枚数）、出力チャネル（出力の枚数）
            # 3x3フィルタで64チャネルを出力
            self.W_conv2 = self.weight_variable([3, 3, 50, 100])
            self.b_conv2 = self.bias_variable([100])
            self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)

            ### 4層目 プーリング層
            self.h_pool2 = self.max_pool_2x2(self.h_conv2)
            # 次の層の畳み込み処理を行う前に、paddingを実行。
            tf.pad(self.h_pool2, [[0, 0], [1, 1], [1, 1], [0, 0]])

            ###5層目 畳み込み層
            self.W_conv3 = self.weight_variable([3, 3, 100, 100])
            self.b_conv3 = self.bias_variable([100])
            self.h_conv3 = tf.nn.relu(self.conv2d(self.h_pool2, self.W_conv3) + self.b_conv3)

            ### 6層目 プーリング層
            self.h_pool3 = self.max_pool_2x2(self.h_conv3)

            tf.pad(self.h_pool3, [[0, 0], [1, 1], [1, 1], [0, 0]])

            ###7層目　畳み込み層
            self.W_conv4 = self.weight_variable([3, 3, 100, 100])
            self.b_conv4 = self.bias_variable([100])
            self.h_conv4 = tf.nn.relu(self.conv2d(self.h_pool3, self.W_conv4) + self.b_conv4)
            self.h_pool4 = self.max_pool_2x2(self.h_conv4)

            tf.pad(self.h_pool4, [[0, 0], [1, 1], [1, 1], [0, 0]])

            ### 全結合層
            # 全結合層にするために、スカラーのテンソルに変形。画像サイズ縦と画像サイズ横とチャネル数の積の次元
            # 出力は1000
            self.weight_fully_connected1 = self.weight_variable([23 * 23 * 100, 1000])  # 44100
            self.bias_fc1 = self.bias_variable([1000])
            # -1 is inferred to be 9:
            # reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
            #         [4, 4, 4, 5, 5, 5, 6, 6, 6]]
            # print "h_pool2"#23*23*100
            # print h_pool2
            self.h_pool4_flat = tf.reshape(self.h_pool4, [-1, 23 * 23 * 100])
            print "h_pool_flat"  # 44100
            print self.h_pool4_flat
            self.hidden_fully_connect1 = tf.nn.relu(tf.matmul(self.h_pool4_flat, self.weight_fully_connected1) + self.bias_fc1)
            ### 6層目 Softmax Regression層

            self.weight_fully_connected2 = self.weight_variable([1000, 361])
            self.bias_fc2 = self.bias_variable([361])

            self.y_conv = tf.nn.softmax(tf.matmul(self.hidden_fully_connect1, self.weight_fully_connected2) + self.bias_fc2)
        self.cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y_conv))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()

        self.init = tf.initialize_all_variables()

    def search_deep_learning(self, sess, go_state_obj, current_player):
        xTrain = self.make_input.generate_input(go_state_obj, current_player)
        output_array=sess.run(self.y_conv, feed_dict={self.x_input: xTrain})
        #two_dimensional_array = tf.reshape(output_array, [-1, 19, 19, 1])
        return output_array

    letter_coords = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']

    def coord_to_str(self,row, col):
        return self.letter_coords[row] + str(col + 1)  # 数字をcharacterに変換
    def gtp_io(self,sess):
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
        print "19路盤専用！"

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
                    sys.stderr.write(state.print_board(go_state_obj))
                else:
                    if command[1][0] == "b":
                        player = players[0]
                    elif command[1][0] == "w":
                        player = players[1]

                    x = self.letter_coords.index(command[2][0].upper())  # リストは0からスタート
                    y = 18 - (int(command[2][1:]) - 1)

                    go_state_obj = state.move(go_state_obj, player, (x, y))
                    print player.player_id
                    # print go_state_obj._board
                    # print state.print_board(go_state_obj)

                ret = "= \n\n"

            elif command[0] == "genmove":
                if command[1] == "b":
                    player = players[0]
                elif command[1] == "w":
                    player = players[1]
                (go_state_obj, move) = search_algorithm.next_move(self,sess,go_state_obj, player)
                # tup=[move[0],move[1]]
                if move == "pass":
                    ret = "pass"
                else:
                    ret = self.coord_to_str(move[0], 18 - move[1])
                    sys.stderr.write("move x:" + str(move[0]) + "\n")
                    sys.stderr.write("move: y" + str(move[1]) + "\n")
                    go_state_obj = state.move(go_state_obj, player, move)
                    sys.stderr.write(state.print_board(go_state_obj))

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
                break
            else:
                # print 'Warning: Ignoring unknown command'
                pass
            print ret
            sys.stdout.flush()
    def test(self,sess):
        players = [Player(0.0, 'human'), Player(1.0, 'human')]
        players[0].next_player = players[1]
        players[1].next_player = players[0]
        rule = Go()

        player = players[0]
        go_state_obj=GoStateObject()

        xTrain = self.make_input.generate_input(go_state_obj, player)
        print sess.run(self.y_conv, feed_dict={self.x_input: xTrain})
        return
    def _main(self):
        with tf.Session() as sess:
            self.saver.restore(sess, "model.ckpt")  # モデルの読み込み　作業の再開時にコメントアウト
            self.test(sess)
            #self.test(sess)
            print "gtp_io"
            self.gtp_io(sess)
