#!/usr/bin/env python
#coding:utf-8
#python2.7

import pyximport

pyximport.install()

import sgf  # Please install "sgf 0.5". You can install it by using the command of "pip install sgf==0.5".
import re
import os  # osモジュールのインポート
from game import Player

from go import Go
from go import GoStateObject
import sys
import tensorflow as tf
sys.setrecursionlimit(1000000)
# sysモジュールをリロードする
reload(sys)
# デフォルトの文字コードを変更する
sys.stderr.write(str(sys.setdefaultencoding('utf-8')))
# デフォルトの文字コードを出力する

def get_symbol(player):
    symbols = '*o'
    return symbols[player.player_id]

'''
def read_move(rule, player):

  #:param state: 現在の状態オブジェクト
  #:param player: 現在のプレーヤー
  #:return: 読み込んだ手の座標を返すmove = (x, y)

  while True:
    xy = raw_input('Enter your move for %s (e.g., 3,5) or enter pass: ' % get_symbol(player))
    try:
      if xy == "pass":#passの処理
        return 3
      x, y = map(int, xy.split(','))
      move = (x, y)
      if rule.valid_move(player, move):
        return move
      else:
        print "try again"
    except:
      print "try again"
      pass

'''
letter_coords = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
def coord_to_str(row, col):
    return letter_coords[row] + str(col + 1)  #数字をcharacterに変換
def gtp_io():
    players = [Player(0, 'human'), Player(1, 'human')]
    players[0].next_player = players[1]
    players[1].next_player = players[0]
    player = players[0]
    rule = Go()
    go_state_obj = GoStateObject()
    #search_algorithm = SimpleSearch()
    #from search import MontecarloSearch
    #search_algorithm = MontecarloSearch()
    from search import DeepLearningMontecarlo
    search_algorithm = DeepLearningMontecarlo()
    """ Main loop which communicates to gogui via GTP"""
    known_commands = ['boardsize', 'clear_board', 'komi', 'play', 'genmove',
                      'final_score', 'quit', 'name', 'version', 'known_command',
                      'list_commands', 'protocol_version', 'gogui-analyze_commands']
    analyze_commands = ["gfx/Predict Final Ownership/predict_ownership",
                        "none/Load New SGF/loadsgf"]
    #print("starting main.py: loading %s" %sgf_file,file=sys.stderr)
    #output_file = open("output.txt", "wb")
    #output_file.write("intializing\n")
    print "19路盤専用！"

    while True:
        try:
            line = raw_input().strip()
            #print line
            #output_file.write(line + "\n")
        except EOFError:
            #output_file.write('Breaking!!\n')
            break
        if line == '':
            continue
        command = [s.lower() for s in line.split()]
        #print command
        if re.match('\d+', command[0]):
            command = command[1:]

        ret = ''
        if command[0] == "clear_board":
            rule = Go()
            go_state_obj = GoStateObject()

            ret = "= \n\n"
        elif command[0] == "komi":
            ret = "= \n\n"
        elif command[0] == "play":  #playにバグあり "play w C19"が渡された時はx座標,yの順番
            if command[2] == "pass":
                sys.stderr.write(rule.print_board(go_state_obj))
            else:
                if command[1][0] == "b":
                    player = players[0]
                elif command[1][0] == "w":
                    player = players[1]

                x = letter_coords.index(command[2][0].upper())  #リストは0からスタート
                y = 18 - (int(command[2][1:]) - 1)

                go_state_obj = rule.move(go_state_obj, player, (x, y))
                print player.player_id
                #print go_state_obj._board
                #print rule.print_board(go_state_obj)

            ret = "= \n\n"

        elif command[0] == "genmove":
            if command[1] == "b":
                player = players[0]
            elif command[1] == "w":
                player = players[1]
            (go_state_obj, move) = search_algorithm.next_move(go_state_obj, player)
            #tup=[move[0],move[1]]
            if move == "pass":
                ret = "pass"
            else:
                ret = coord_to_str(move[0], 18 - move[1])
                sys.stderr.write("move x:" + str(move[0]) + "\n")
                sys.stderr.write("move: y" + str(move[1]) + "\n")
                sys.stderr.write(rule.print_board(go_state_obj))
                print "turns_since_board"
                print rule.print_board(go_state_obj)
                print go_state_obj._turns_since_board
            ret = '= ' + ret + '\n\n'

        elif command[0] == "final_score":
            #print("final_score not implemented", file=sys.stderr)
            ret = "= \n\n"
        elif command[0] == "name":
            ret = '= EsperanzaGo\n\n'
        elif command[0] == "predict_ownership":
            #ownership_prediction = driver.evaluate_current_board()
            #ret = influence_str(ownership_prediction)
            ret = "= \n\n"
        elif command[0] == "version":
            ret = '= 1.0\n\n'
        elif command[0] == "list_commands":
            #ret = '= \n'.join(known_commands)
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
            #ret = '= \n\n'
            #print ret
            #print('=%s \n\n' % (cmdid,), end='')
            break
        else:
            #print 'Warning: Ignoring unknown command'
            pass
        print ret
        sys.stdout.flush()
'''
def human_play_mode():
  players = [Player(0, 'human'), Player(1, 'human')]
  players[0].next_player = players[1]
  players[1].next_player = players[0]
  player = players[0]
  rule = Go()
  go_state_obj=GoStateObject()
  search_algorithm = MiniMaxSearch(1)
  rule.print_board(go_state_obj)
  while True:
    print rule.next_moves(go_state_obj,player)
    if player.player_name == 'human':
      go_state_obj = rule.move(player, read_move(rule, player))
    else:
      (go_state_obj, move) = search_algorithm.next_move(go_state_obj, player)
      print "%s's move for %s: %s" % (str(player).title(),get_symbol(player),move)
      go_state_obj = rule.move(players[1], (move[0],move[1]))
    rule.print_board(go_state_obj)
    #if state.win(player):
    #  print '%s wins!' % str(player).title()
    #  break
    #elif state.win(player.next()):
    #  print '%s wins!' % str(player.next()).title()
    #  break
    #elif state.draw(player):
    #  print 'Draw'
    #  break
    #else:
    player = player.next()
'''
character_list = [chr(i) for i in range(97, 97 + 26)]
def sgf_mode():
    players = [Player(0.0, 'human'), Player(1.0, 'human')]
    players[0].next_player = players[1]
    players[1].next_player = players[0]
    #player = players[0]
    state = Go()
    # 指定したパス内の全てのファイルとディレクトリを要素とするリストを返す
    #自分の環境でのsgf_filesへのパスを書く。
    files = os.listdir(os.getcwd() + "/sgf_files")

    for file_name in files:
        print file_name

        with open("sgf_files/" + file_name) as f:  #ディレクトリEsperanzaGoからのsgf_filesへの相対パス
            #with open("sgf_files/*") as f:
            collection = sgf.parse(f.read())
            flag = False
            for game in collection:

                for node in game:
                    if flag == False:
                        flag = True
                        continue
                    print node.properties
                    #print node.next
                    #print state
                    lists = node.properties.values()
                    print lists
                    internal_lists = lists[0]
                    position = internal_lists[0]
                    x = character_list.index(position[0])
                    y = character_list.index(position[1])
                    print x, y
                    if node.properties.has_key('B') == True:
                        state = state.move(players[0], (x, y))
                    elif node.properties.has_key('W') == True:
                        state = state.move(players[1], (x, y))
                    print state

def terminal_mode():
    players = [Player(0.0, 'human'), Player(1.0, 'human')]
    players[0].next_player = players[1]
    players[1].next_player = players[0]
    player = players[0]
    rule = Go()
    go_state_obj = GoStateObject()
    #search_algorithm = SimpleSearch()
    search_algorithm = MontecarloSearch()
    while True:
        try:
            line = raw_input().strip()
            #print line
            #output_file.write(line + "\n")
        except EOFError:
            #output_file.write('Breaking!!\n')
            break
        if line == '':
            continue
        command = [s.lower() for s in line.split()]
        #print command
        if re.match('\d+', command[0]):
            command = command[1:]

        ret = ''
        if command[0] == "clear_board":
            rule = Go()
            ret = "= \n\n"
        elif command[0] == "play":  #playにバグあり "play w C19"が渡された時はx座標は,yの順番
            if command[2] == "pass":
                rule.print_board(go_state_obj)
            else:
                if command[1][0] == "b":
                    player = players[0]
                elif command[1][0] == "w":
                    player = players[1]

                x = letter_coords.index(command[2][0].upper())  #リストは0からスタート
                y = 18 - (int(command[2][1:]) - 1)
                go_state_obj = rule.move(go_state_obj, player, (x, y))
                print rule.print_board(go_state_obj)
        elif command[0] == "genmove":
            if command[1] == "b":
                player = players[0]
            elif command[1] == "w":
                player = players[1]
            #flag for next_move
            (go_state_obj, move) = search_algorithm.next_move(go_state_obj, player)
            print rule.print_board(go_state_obj)

            if move == "pass":
                pass
            else:
                go_state_obj = rule.move(go_state_obj, player, move)
                print rule.print_board(go_state_obj)

if __name__ == '__main__':
    # #If you want to use GoGUI,please delete the comment tag of this line.
    #gtp_io()
    #GoGUIを使わないモードの場合は以下を有効にする。
    '''
    from forward_prop_network import ForwardPropNetwork
    with tf.Session() as sess:
        ForwardPropNetwork(sess)
    '''


    print "注意！　go_state_objをfor文などで使うときはcopy.deepcopy()を使用すること。"
    print "Please execute cython by following command. python kifu/kiati-go/CythonGo/main.pyx"
    mode = raw_input('Enter mode \n1:GTP_MODE\n2:SGFmode\n3:Montecarlo(terminal_mode)\n4:Train Network from sgf\n5:ForwardProp\n6:make Pickle files\n7:test forward prop\n8:value network train\n9:MID network Train\n10:Policy 5x5 Network Train\n11:Value Network Train from LABELs\n12:Fast Policy Network started\n13:train_policy_batchnorm_4layer\n14:13layer Large policy\n15:9layer Large policy\n16:8Layer policy legal\n17:NORMAL policy legal')

    if mode == "0":
        #human_play_mode()
        pass
    elif mode == "1":
        #from search import MontecarloSearch

        #GoGUIで対戦するときに使うモードのテスト
        gtp_io()
    elif mode == "2":
        sgf_mode()
    elif mode == "3":
        from search import MontecarloSearch
        terminal_mode()
    elif mode == "4":
        print "Please set board size as 19."
        #sess=tf.Session()
        from train_network import Train
        Train()
    elif mode == "5":
        print "Please set board size as 19"
        from forward_prop_network import ForwardPropNetwork
        with tf.Session() as sess:
            ForwardPropNetwork(sess)

    elif mode == "6":
        print "shuffle and make pickle files Warning! The data might be large."
        #Pickle()
    elif mode == "7":
        from forward_prop_network_test import ForwardPropTest
        print "test forward prop"
        ForwardPropTest()
    elif mode == "8":
        print "Value Network Train"
        sess=tf.Session()
        from value_network import ValueNetwork

        ValueNetwork(sess)
    elif mode =="9":
        print "Please set board size as 19."
        from mid_speed_policy import MidSpeedPolicyTrain
        #sess=tf.Session()
        MidSpeedPolicyTrain()
    elif mode =="10":
        print "5x5 train"
        from train_network_5x5 import Train5x5
        Train5x5()
    elif mode == "11":
        print "value_network_with_labels training"
        from value_network_with_label import ValueNetworkWithLabel
        sess=tf.Session()

        ValueNetworkWithLabel(sess)
    elif mode == "12":
        from train_network_Fast import FastPolicyNetwork
        FastPolicyNetwork()
    elif mode == "13":
        from train_network_batchnorm_4layer import TrainBatchnorm4layer
        TrainBatchnorm4layer()
    elif mode == "14":
        from train_network_13layer_batch import Train13Layer
        Train13Layer()
    elif mode == "15":
        from train_network_9layer_batch import Train9Layer
        print "Train9Layer Batch Norm version (not recommended)"
        Train9Layer()
    elif mode == '16':
        from train_network_8layer_policy_legal import Train8LayerPolicyLegal
        Train8LayerPolicyLegal()
    elif mode == '17':
        from train_network_normal import Train
        Train()