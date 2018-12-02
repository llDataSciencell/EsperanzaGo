# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 09:56:26 2017

@author: tomohiro
"""

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
import pickle as pickle
import random
import sys
# sysモジュールをリロードする
reload(sys)
# デフォルトの文字コードを変更する
sys.setdefaultencoding('utf-8')
# デフォルトの文字コードを出力する
print 'defaultencoding:', sys.getdefaultencoding()
#import input_data#delete
#パスはどうする？forwardした結果一番良い答えがパスかもしれない
'''input_planeが変更になる可能性があり、データサイズも大きくなってしまうので
go_state_objとplayerのみを保存することにする。'''
class Pickle(GoVariable):
        character_list=[chr(i) for i in range(97, 97+26)]
        def __init__(self):
            #self.train()
            self.train()
        def train(self):
          players = [Player(0.0, 'human'), Player(1.0, 'human')]
          players[0].next_player = players[1]
          players[1].next_player = players[0]
          #player = players[0]
          rule = Go()
          # 指定したパス内の全てのファイルとディレクトリを要素とするリストを返す
          #自分の環境でのsgf_filesへのパスを書く。
          pt=os.getcwd()+u"/sgf/"
          dirnames = os.listdir(pt)




          #saver.restore(sess, "model.ckpt")＃モデルの読み込み　作業の再開時にコメントアウト
          make_input=MakeInputPlane()
          pickle_obj=[]

          #pickle.dump('Hello, World!', f, protocol=2)
          total_file_num=0
          for dir1 in dirnames:
              path1=pt+dir1
              files=os.listdir(path1)
              total_file_num+=len(files)

          num=0
          for dir1 in dirnames:
                   path1=pt+dir1
                   files=os.listdir(path1)

                   #random_used_num=[index for index in xrange(total_file_num*230)]
                   print "len of files"
                   print len(files)

                   for filenm in files:
                        pickle_boards=[]
                        file_name=path1+"/"+filenm
                        #file_name = os. listdir(path2)
                        with open(file_name) as f: #ディレクトリEsperanzaGoからのsgf_filesへの相対パス
                          #with open("sgf_files/*") as f:
                          go_state_obj=GoStateObject()
                          collection = sgf.parse(f.read())
                          flag=False
                          for game in collection:
                            for node in game:
                              if flag==False:
                                flag=True
                                continue
                              print node.properties
                              print node.next

                              lists=node.properties.values()
                              #print lists
                              try:
                                  internal_lists=lists[0]
                                  position=internal_lists[0]

                                  xpos=self.character_list.index(position[0])
                                  ypos=self.character_list.index(position[1])

                                  print xpos,ypos
                                  if node.properties.has_key('B')==True:
                                      current_player=players[0]
                                  elif node.properties.has_key('W')==True:
                                      current_player=players[1]
                                  go_state_obj = rule.move(go_state_obj,current_player, (xpos,ypos))


                              except:
                                  continue

                              #input_data=make_input.generate_input(go_state_obj,current_player)
                              #output_data=make_input.generate_answer(pos_tuple)

                              #num=random.choice(random_used_num)
                              #random_used_num.remove(num)
                              pickle_obj=[go_state_obj,current_player]
                              pickle_boards.extend(pickle_obj)
                              num+=1
                          pt=os.getcwd()+"/kifu_pickle/"+num+".pck"
                          with open(pt, mode='wb') as f:
                              #pickle_obj.append([input_data,output_data]
                              pickle.dump(pickle_boards, f, protocol=2)
