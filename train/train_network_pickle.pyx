#coding:utf-8
#TODO: batchでまんべんなくデータが含まれるようにする。
#TODO: オブジェクトだからこのファイルを実行しても何も起きないので、mainから実行する。
#sgf読み込みで新しい棋譜になったら盤面を初期化する
import sgf  # Please install "sgf 0.5". You can install it by using the command of "pip install sgf".
import sys
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
import sys
# sysモジュールをリロードする
reload(sys)
# デフォルトの文字コードを変更する
sys.stderr.write(str(sys.setdefaultencoding('utf-8')))
#import input_data#delete
#パスはどうする？forwardした結果一番良い答えがパスかもしれない
class TrainFromPickle(GoVariable):
    character_list=[chr(i) for i in range(97, 97+26)]
    def __init__(self):
      #self.train()
      self.train()
    def weight_variable(self,shape):
        """適度にノイズを含んだ（対称性の除去と勾配ゼロ防止のため）重み行列作成関数
        """

        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        """バイアス行列作成関数
        """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self,x, W):
        """2次元畳み込み関数
        """
        return tf.nn.conv2d(x,
                            W,
                            strides=[1, 1, 1, 1], # 真ん中2つが縦横のストライド 1,1で画像が小さくならない
                            padding='SAME')

    def max_pool_2x2(self,x):
        """2x2マックスプーリング関数
        """

        return tf.nn.max_pool(x,ksize=[1, 1, 1, 1],strides=[1, 1, 1, 1],padding='SAME')
    def train(self):
          players = [Player(0.0, 'human'), Player(1.0, 'human')]
          players[0].next_player = players[1]
          players[1].next_player = players[0]
          #player = players[0]
          rule = Go()

          # mnistのダウンロード
          #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
          print "starting tf.device(/gpu:1)"
          sess = tf.InteractiveSession()
          #with tf.device("/gpu:1"):
          with tf.device("/cpu:0"):
                #5290000=23*23*100*100  88200=21*21*
                # データ用可変2階テンソルを用意
                x_input = tf.placeholder("float", shape=[None,6,361])
                # 正解用可変2階テンソルを用意
                y_ = tf.placeholder("float", shape=[None,361])

                # 画像を2次元配列にリシェイプ 第1引数は画像数(-1は元サイズを保存するように自動計算)、縦x横、チャネル
                x_image = tf.reshape(x_input, [-1,19, 19, 6])

                print x_image

                x_image_pad=tf.pad(x_image,[[0,0],[2,2],[2,2],[0,0]])


                ### 1層目 畳み込み層
                ### 1層目 畳み込み層

                # 畳み込み層のフィルタ重み、引数はパッチサイズ縦、パッチサイズ横、入力チャネル数、出力チャネル数
                # 5x5フィルタで100チャネルを出力（入力は白黒画像なので1チャンネル）
                W_conv1 = self.weight_variable([5, 5, 6, 50])
                # 畳み込み層のバイアス
                b_conv1 = self.bias_variable([50])
                # 活性化関数ReLUでの畳み込み層を構築
                h_conv1 = tf.nn.relu(self.conv2d(x_image_pad, W_conv1) + b_conv1)
                #4次元のpaddingをする
                ### 2層目 プーリング層
                # 2x2のマックスプーリング層を構築
                h_pool1 = self.max_pool_2x2(h_conv1)
                #次の層の畳み込み処理を行う前に、paddingを実行。
                tf.pad(h_pool1,[[0,0],[1,1],[1,1],[0,0]])

                ### 3層目 畳み込み層

                # パッチサイズ縦、パッチサイズ横、入力チャネル（枚数）、出力チャネル（出力の枚数）
                # 3x3フィルタで64チャネルを出力
                W_conv2 = self.weight_variable([3, 3, 50, 100])
                b_conv2 = self.bias_variable([100])
                h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)


                ### 4層目 プーリング層
                h_pool2 = self.max_pool_2x2(h_conv2)
                #次の層の畳み込み処理を行う前に、paddingを実行。
                tf.pad(h_pool2,[[0,0],[1,1],[1,1],[0,0]])

                ###5層目 畳み込み層
                W_conv3 = self.weight_variable([3, 3, 100, 100])
                b_conv3 = self.bias_variable([100])
                h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3) + b_conv3)


                ### 6層目 プーリング層
                h_pool3 = self.max_pool_2x2(h_conv3)

                tf.pad(h_pool3,[[0,0],[1,1],[1,1],[0,0]])

                ###7層目　畳み込み層
                W_conv4 = self.weight_variable([3, 3, 100, 100])
                b_conv4 = self.bias_variable([100])
                h_conv4 = tf.nn.relu(self.conv2d(h_pool3, W_conv4) + b_conv4)
                h_pool4 = self.max_pool_2x2(h_conv4)

                tf.pad(h_pool4,[[0,0],[1,1],[1,1],[0,0]])

                ### 全結合層
                # 全結合層にするために、スカラーのテンソルに変形。画像サイズ縦と画像サイズ横とチャネル数の積の次元
                # 出力は1000
                weight_fully_connected1 = self.weight_variable([23*23*100,1000])#44100
                bias_fc1 = self.bias_variable([1000])
                # -1 is inferred to be 9:
                #reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                #         [4, 4, 4, 5, 5, 5, 6, 6, 6]]
                #print "h_pool2"#23*23*100
                #print h_pool2
                h_pool4_flat = tf.reshape(h_pool4, [-1, 23*23*100])
                print "h_pool_flat"#44100
                print h_pool4_flat
                hideen_fully_connect1 = tf.nn.relu(tf.matmul(h_pool4_flat, weight_fully_connected1) + bias_fc1)

                # ドロップアウトを指定
                keep_prob = tf.placeholder("float")
                h_fc1_drop = tf.nn.dropout(hideen_fully_connect1, keep_prob)

                ### 6層目 Softmax Regression層

                weight_fully_connected2 = self.weight_variable([1000, 361])
                bias_fc2 = self.bias_variable([361])

                y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, weight_fully_connected2) + bias_fc2)
          cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
          train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
          correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
          accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
          sess.run(tf.initialize_all_variables())
          saver = tf.train.Saver()

          # 指定したパス内の全てのファイルとディレクトリを要素とするリストを返す
          #自分の環境でのsgf_filesへのパスを書く。
          path1 = os.listdir(os.getcwd()+"/kifu_pickle")


          batch_count_num=0
          train_count_num=0
          init = tf.initialize_all_variables()
          xTrain=[]
          yTrain=[]
          print "aaa"


          total_file_num=len(path1)

          print "total file num:"+str(total_file_num)

          with tf.Session() as sess:
                sess.run(init)#If it is first time of learning
                #saver.restore(sess, "model.ckpt")＃モデルの読み込み　作業の再開時にコメントアウト
                make_input=MakeInputPlane()

                #with open("sgf_files/*") as f:
                #go_state_obj=GoStateObject()
                for _ in xrange(1000):
                    try:
                          train_count_num+=1

                          pck_path=os.getcwd()+"/kifu_pickle/"+str(random.randint(total_file_num-1))+".pck"

                          with open(pck_path, mode='rb') as f:
                            pickle_boards=pickle.load(f)

                          [go_state_obj,current_player,pos_tuple] = pickle_boards[random.randint(0,len(pickle_boards)-1)]
                          xTrain.append(make_input.generate_input(go_state_obj,current_player))
                          yTrain.append(make_input.generate_answer(pos_tuple))
                          #xTrain,yTrainの中身を代入する処理をここに書く。
                          batch_count_num+=1

                          if batch_count_num==50:
                            #print xTrain
                            train_step.run(feed_dict={x_input:xTrain, y_: yTrain, keep_prob: 0.5})
                            batch_count_num=0
                            xTrain=[]
                            yTrain=[]
                          print "train_count_num"
                          print train_count_num
                          if train_count_num > 10000:
                              train_count_num = 0

                              saver.save(sess,"model.ckpt")
                    except:
                          pass
