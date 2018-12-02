#coding:utf-8
import random
import copy
import math
from go import GoVariable
from go import Go
from go import GoStateObject
import random
import numpy as np
import sys
#import tensorflow as tf
import traceback
import tensorflow as tf
import random
import threading
import time

# sysモジュールをリロードする
reload(sys)
# デフォルトの文字コードを変更する
sys.stderr.write(str(sys.setdefaultencoding('utf-8')))

'''パスはどうする？forwardで求めた答えで一番良いのがパスかもしれない'''

'''SimpleSearchに不具合ができてしまったので修正する！'''
class Search(object):
  def __init__(self, *args, **kwargs):
    raise NotImplementedError

  def next_move(self, state, player):
    '''Returns (next_state, next_move) tuple.'''
    raise NotImplementedError

'''
class NodeTree(GoVariable):
  def __init__(self):
    self.nodes=[]
'''
class DeepLearningSearch(GoVariable):
  def __init__(self):
    self.rule=Go()
  def two_dimensional_index(self,pos_num):
    y=int(math.floor(pos_num / self.rule._SIZE))
    x=int(pos_num - y * self.rule._SIZE)
    return (x,y)
  #don't delete the train argument. train is passed by forward_prop_network.py by (self,.....) train is forward_prop's self
  def next_move(self,train,sess,go_state_obj, player,last_opponent_move):
    output_board=train.search_deep_learning(sess,go_state_obj,player)

    list_board=output_board.tolist()
    #sys.stderr.write(str(list_board))
    next_move_list=self.rule.next_moves_flat(go_state_obj,player,True)
    #sys.stderr.write(str(next_move_list))

    for y in xrange(self._SIZE):
      for x in xrange(self._SIZE):
        if go_state_obj.turns_num < 200:
          if x == 0 or y == 0 or x==self._SIZE-1 or y == self._SIZE -1:
            list_board[0][y*self._SIZE+x]=float(list_board[0][y*self._SIZE+x]/10.0)
          if x == 1 or y == 1 or x==self._SIZE-2 or y == self._SIZE -2:
            list_board[0][y*self._SIZE+x]=float(list_board[0][y*self._SIZE+x]/2.0)
        if go_state_obj.turns_num >= 200 and go_state_obj.turns_num <= 300 :
          if x == 0 or y == 0 or x==self._SIZE-1 or y == self._SIZE -1:
            list_board[0][y*self._SIZE+x]=float(list_board[0][y*self._SIZE+x]/4.0)
        if [y*self._SIZE+x] not in next_move_list:
          list_board[0][y*self._SIZE+x]=None
    flat_move=list_board[0].index(max(list_board[0]))

    (x,y)=self.two_dimensional_index(flat_move)

    if self.rule.valid_move_public(go_state_obj,player,(x,y)):
      next_move_pos = (x,y)
    else:
      print "DeepLearning pass"
      next_move_pos=self.rule._PASS

    if last_opponent_move == self.rule._PASS:
        print "PASS! PASS! PASS! opponent passes"
        rnd=random.random()
        if rnd >= 0.5:
            next_move_pos =self.rule._PASS

    return (self.rule.move_and_return_state(go_state_obj, player, next_move_pos), next_move_pos)

class PredictionValue(GoVariable):
  def __init__(self):
    self.rule=Go()
  def two_dimensional_index(self,pos_num):
    y=int(math.floor(pos_num / self.rule._SIZE))
    x=int(pos_num - y * self.rule._SIZE)
    return (x,y)
  #don't delete the train argument. train is passed by forward_prop_network.py by (self,.....) train is forward_prop's self
  def next_value(self,train,sess,go_state_obj, player):
    output_board=train.search_deep_learning(sess,go_state_obj,player)

    list_board=output_board.tolist()
    #sys.stderr.write(str(list_board))
    next_move_list=self.rule.next_moves_flat(go_state_obj,player,True)
    #sys.stderr.write(str(next_move_list))

    for y in xrange(self._SIZE):
      for x in xrange(self._SIZE):
        if ([y*self._SIZE+x] not in next_move_list):
          list_board[0][y*self._SIZE+x]=-10000
    value_board = list_board[0]

    return value_board


class Node(GoVariable):
  def __init__(self,go_state_obj):
    self.child_num=0
    self.child=[]#child node
    self.child_games_sum=0
    self.go_state_obj=go_state_obj

class Child(GoVariable):
  def __init__(self, move,player):
    self.move = move
    self.player=player
    self.games = 0
    self.rate = 0.0
    self.next_node = self._EMPTY#caution!

class MontecarloSearch(Search,GoVariable):
  def __init__(self):
    self._player = None
    self._ai_player = None
    self._uct_loop=10#The number of trying uct search
    self.NoneChild = Child(None,None)
    self.NoneMove = (-1,-1)
    self.NoneState = Go()
    self.rule=Go()
    self.black_pattern=[]
    self.white_pattern=[]
    from pattern_database import Pattern
    pattern_obj=Pattern()

    self.black_pattern=self.get_expand_pattern_black(pattern_obj.pattern)
    self.white_pattern=self.get_expand_pattern_white(pattern_obj.pattern)
    self.edge_pattern_black =self.get_expand_pattern_black(pattern_obj.edge_pattern)
    self.edge_pattern_white =self.get_expand_pattern_white(pattern_obj.edge_pattern)

    from forward_prop_network import ForwardPropNetwork
    #self.search_algorithm=DeepLearningSearch()
    self.sess=tf.Session()
    #self.forward_prop_network=ForwardPropNetwork(self.sess)

    self.first_node_prediction_obj=PredictionValue()
    self.first_node_prediction_value=None

    #search_algorithm = DeepLearningSearch()
    self.sess = tf.Session()
    self.forward_large_network = ForwardPropNetwork(self.sess)

    self.node =None
    self.threads = []

  letter_kind = GoStateObject()
  _=letter_kind._EMPTY
  B=float(2)#state is outside of board
  A=float(777)#any kind of state is ok
  PATTERN_SIZE=2
  BLOCK_SIZE = 3
  VALUE=1
  def edge_pattern_match(self,board,player,move):
    #print "EDGE passed!!!!!!!!!!!!!!!!!"
    (x,y)=move
    x=int(x)
    y=int(y)
    B=float(2)
    ### get block ###
    if x == 0 and y == 0:
        block = [[B           ,B         ,   B],
                [      B    ,board[y][x],board[y][x+1]],
                [      B    ,board[y+1][x],board[y+1][x+1]]]
    elif x == self.rule._SIZE-1 and y == self.rule._SIZE-1:
        block = [[board[y-1][x-1]  ,board[y-1][x],    B],
                 [board[y][x-1],board[y][x],     B     ],
                 [B            ,       B   ,    B      ]]
    elif x == self.rule._SIZE -1 and y ==0:
        block = [[B ,         B  ,         B],
            [board[y][x-1],board[y][x],    B],
            [board[y+1][x-1],board[y+1][x],B]]
    elif x == 0 and y ==self.rule._SIZE-1:
        block = [[B ,  board[y-1][x]  ,board[y-1][x+1]],
                 [B,   board[y][x],     board[y][x+1]],
                 [B,             B,                   B]]
    elif x == 0:
        block = [[B  ,board[y-1][x]  ,board[y-1][x+1]],
                [B,   board[y][x],board[y][x+1]],
                [B,    board[y+1][x],board[y+1][x+1]]]
    elif y == 0:
        block = [[B        , B        ,         B    ],
                [board[y][x-1],board[y][x],board[y][x+1]],
                [board[y+1][x-1],board[y+1][x],board[y+1][x+1]]]
    elif x == self.rule._SIZE-1:
        block = [[board[y-1][x-1] ,board[y-1][x]  ,B],
                 [board[y][x-1],   board[y][x]  ,  B],
                 [board[y+1][x-1], board[y+1][x],  B]]
    elif y == self.rule._SIZE-1:
        block = [[board[y-1][x-1]  ,board[y-1][x]  ,board[y-1][x+1]],
                 [board[y][x-1],    board[y][x],    board[y][x+1]],
                 [B,                  B,                        B]]

    ### pattern matching ###
    if player.player_id == 0:
        for pat_num in xrange(0,self.PATTERN_SIZE):
            try:
                for xpos in range(self.BLOCK_SIZE):
                    for ypos in range(self.BLOCK_SIZE):
                        if self.edge_pattern_black[pat_num][0][ypos][xpos]==self.A or block[ypos][xpos] == self.edge_pattern_black[pat_num][0][ypos][xpos]:
                            continue
                        else:
                            #print "exception"
                            raise Exception
                #if all values matchs
                return [self.pattern[pat_num][self.VALUE], (x,y), pat_num]
            except:
                pass
        return None
    elif player.player_id == 1:
        for pat_num in xrange(0,self.PATTERN_SIZE):
            try:
                for xpos in range(self.BLOCK_SIZE):
                    for ypos in range(self.BLOCK_SIZE):
                        if self.edge_pattern_white[pat_num][0][ypos][xpos]==self.A or block[ypos][xpos] == self.edge_pattern_white[pat_num][0][ypos][xpos]:
                            continue
                        else:
                            raise Exception
                #if all values matchs
                return [self.pattern[pat_num][self.VALUE], (x,y), pat_num]
            except:
                pass
        return None
    else:
        print "player is not black or white"
        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit(0)
  def pattern_match(self,board,player,move):
    (x,y)=move
    ### get block ###
    block = [[board[y-1][x-1]  ,board[y-1][x]  ,board[y-1][x+1]],
            [board[y][x-1],board[y][x],board[y][x+1]],
            [board[y+1][x-1],board[y+1][x],board[y+1][x+1]]]

    ### pattern matching ###


    if player.player_id == 0:
        for pat_num in xrange(0,self.PATTERN_SIZE):
            try:
                for xpos in range(self.BLOCK_SIZE):
                    for ypos in range(self.BLOCK_SIZE):
                        if self.black_pattern[pat_num][0][ypos][xpos] == self.A or block[ypos][xpos] == self.black_pattern[pat_num][0][ypos][xpos]:
                            continue
                        else:
                            #print "exception"
                            raise Exception
                #if all values matchs
                return [self.pattern[pat_num][self.VALUE], (x,y), pat_num]
            except:
                pass
        return None
    elif player.player_id == 1:
        for pat_num in xrange(0,self.PATTERN_SIZE):
            try:
                for xpos in range(self.BLOCK_SIZE):
                    for ypos in range(self.BLOCK_SIZE):
                        if self.white_pattern[pat_num][0][ypos][xpos] == self.A or block[ypos][xpos] == self.white_pattern[pat_num][0][ypos][xpos]:
                            continue
                        else:
                            raise Exception
                #if all values matchs
                return [self.pattern[pat_num][self.VALUE], (x,y), pat_num]
            except:
                pass
        return None
    else:
        print "player is not black or white"
        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit(0)

  ### expand patterns ###
  #p:pattern
  def clockwise(self,p):
      #make black version and white version
      q = [[[p[0][2][0],p[0][1][0],p[0][0][0]],
           [p[0][2][1],p[0][1][1],p[0][0][1]],
           [p[0][2][2],p[0][1][2],p[0][0][2]]],p[1]]

      return q


  def reverse(self,p):
      reverted = [[[p[0][0][2],p[0][0][1],p[0][0][0]],
                 [p[0][1][2],p[0][1][1],p[0][1][0]],
                 [p[0][2][2],p[0][2][1],p[0][2][0]]],p[1]]

      return reverted
  def black_white_exchange(self,pattern):
      exchanged = copy.deepcopy(pattern)
      for i in xrange(0,self.PATTERN_SIZE):
          for j in xrange(0,self.PATTERN_SIZE):
              if pattern[0][i][j] == self.rule._BLACK:
                  exchanged[0][i][j] = self.rule._WHITE
              elif pattern[0][i][j] == self.rule._WHITE:
                  exchanged[0][i][j] == self.rule._BLACK
      return exchanged
  def get_expand_pattern_black(self,pattern_obj):
    expand_pattern = []
    for i in xrange(0,self.PATTERN_SIZE):
          p1=pattern_obj[i]#TODO: Error!
          p2=self.clockwise(p1)
          p3=self.clockwise(p2)
          p4=self.clockwise(p3)
          p5=self.reverse(p4)
          p6=self.clockwise(p5)
          p7=self.clockwise(p6)
          p8=self.clockwise(p7)
          expand_pattern.append(p1)
          expand_pattern.append(p2)
          expand_pattern.append(p3)
          expand_pattern.append(p4)
          expand_pattern.append(p5)
          expand_pattern.append(p6)
          expand_pattern.append(p7)
          expand_pattern.append(p8)
    return expand_pattern
  def get_expand_pattern_white(self,pattern_obj):
    expand_pattern = []
    for i in xrange(self.PATTERN_SIZE):
        p1=self.black_white_exchange(pattern_obj[i])
        p2=self.clockwise(p1)
        p3=self.clockwise(p2)
        p4=self.clockwise(p3)
        p5=self.reverse(p4)
        p6=self.clockwise(p5)
        p7=self.clockwise(p6)
        p8=self.clockwise(p7)
        expand_pattern.append(p1)
        expand_pattern.append(p2)
        expand_pattern.append(p3)
        expand_pattern.append(p4)
        expand_pattern.append(p5)
        expand_pattern.append(p6)
        expand_pattern.append(p7)
        expand_pattern.append(p8)
  def playout(self, init_state, player,root_player,myself):
    init_player=player
    go_state_obj=copy.deepcopy(init_state)
    board_size=self.rule._SIZE
    roop_num=board_size*board_size+50
    #ok_move_list=[]
    before_move=self.NoneMove
    previous_place=self.rule._PASS
    for i in xrange(roop_num):
        if (i + 25) % 10 ==0:
            if myself.is_running == False:
                return -100
        #next_states = [(state.move_and_return_state(copy.deepcopy(state._board),player, move), move)
        #            for move in state.next_moves(player)]
        if before_move != self.rule._PASS:
          (before_x,before_y) = before_move

        empty_list=[]
        if before_move != self.rule._PASS:
          #TODO: for x,yを１手前の石周辺だけにする 最初の一手どこかに石が無いと始まらない
          #for x in xrange(0,self.rule._SIZE):
          #  for y in xrange(0,self.rule._SIZE):
          for x in xrange(before_x-1,before_x+2):
            for y in xrange(before_y-1,before_y+2):
            #for x in xrange(before_x-3,before_x+4):
            #  for y in xrange(before_y-3,before_y+4):
              #TODO: パターンの左上に合わせているので、中心のx+1,y+1におく。
              if x < 0 or y < 0 or x >= self.rule._SIZE or y >= self.rule._SIZE:
                #0 is ok self.rule._SIZE-1 is ok
                continue
              if go_state_obj._board[y][x] != self.rule._EMPTY:
                continue
              if x == 0 or y == 0 or x == self.rule._SIZE-1 or y == self.rule._SIZE-1:
                if self.rule.valid_move_public(go_state_obj,player,(x,y)):
                  em_list_returned=self.edge_pattern_match(go_state_obj._board,player,(x,y))
                  if em_list_returned != None:
                    em_list_returned[0]=em_list_returned[0] /float(10)
                    empty_list.append(em_list_returned)
              elif self.rule.valid_move_public(go_state_obj,player,(x,y)):
                em_list_returned=self.pattern_match(go_state_obj._board,player,(x,y))
                if em_list_returned !=None:
                    if x <= 1 or y <= 1 or x >=self.rule._SIZE -2 or y >= self.rule._SIZE -2:
                        em_list_returned[0] =em_list_returned[0] / float(3)
                    empty_list.append(em_list_returned)
              #if abs(x-before_x) == 1 or abs(y-before_y)==1:
        max_num = -9999
        if len(empty_list) != 0:
          for num in range(0,len(empty_list)):
              if empty_list[num][0] > max_num:
                max_num =empty_list[num][0]
                move = empty_list[1]
        else:
          #もしパターンにマッチしないけど打つ場所がある場合は
          next_moves =self.rule.next_moves(go_state_obj,player,True)
          try:
            move=next_moves[random.randint(0,len(next_moves))]
            if self.rule.valid_move_public(go_state_obj,player,move) == False:
                raise Exception
          except:
            #sys.stderr.write("exception!!!!!!!!!!!! next_moves Error!")
            #traceback.print_exc()
            move=self.rule._PASS

        if move==self.rule._PASS and before_move == self.rule._PASS:
            break;

        go_state_obj=self.rule.move(go_state_obj,player,move)
        #print self.rule.print_board(go_state_obj)
        player = player.next_player
        before_move=move

    score=self.rule.count_territory(go_state_obj)

    score=score[0]-score[1]#black-white

    #If AI player wins,return 1.If Ai lose,return 0.
    if int(root_player.player_id) == int(self._BLACK):
        if score >= 0:
            return 1
        else:
            return 0
    elif int(root_player.player_id) == int(self._WHITE):
        if score >= 0:
            return 0
        else:
            return 1
  def xy_to_flat_index(self,move):
    (x,y)=move
    #pos_num=(x,y)
    #y=int(math.floor(pos_num / self.rule._SIZE))
    #x=int(pos_num - y * self.rule._SIZE)
    flatten_pos = y*19+x
    return flatten_pos
  def create_node(self,go_state_obj,player):
    next_moves = self.rule.next_moves(go_state_obj,player,True)

    node = Node(go_state_obj)
    for move in next_moves:
      node.child.append(Child(move,player))
    node.child.append(Child(self._PASS,player))

    node.go_state_obj = go_state_obj

    return node

  def get_best_ucb(self,node,FIRST_NODE):#node=-1 node.next error
    #全てのボードのマス目の評価値の中で、一番大きいものを選択しているので、探索は満遍なく行える。
    max_ucb=-999999
    best_child = self.NoneChild

    for child in node.child:
      if child.games == 0:
        #ucb=10000000
        if child.move == self.rule._PASS:
            ucb =0.1
        else:
            (x,y)=child.move
            ucb = 1000*self.first_node_prediction_value[self.xy_to_flat_index(child.move)]
            if x <= 0 or y <= 0 or x >=self.rule._SIZE-1 or x >=self.rule._SIZE-1:
                ucb = float(ucb) / float(20)
            elif x ==1 or y == 1 or x==self.rule._SIZE-2 or y==self.rule._SIZE-2:
                ucb = float(ucb) / float(4)
            #UCB VALUE usually 0.0 to 0.7 maximum is 4
            #print "ucb Value:"+str(ucb)
      else:
        C=15
        if FIRST_NODE == True and child.move != self.rule._PASS:
          (x,y)=child.move
          rate_value=child.rate*4 + C * math.sqrt(math.log(node.child_games_sum)/child.games)
          ucb = rate_value + 100*self.first_node_prediction_value[self.xy_to_flat_index(child.move)]
          if x == 0 or y == 0 or x >=self.rule._SIZE-1 or x >=self.rule._SIZE-1:
              ucb = float(ucb) / float(20)
          elif x ==2 or y == 2 or x==self.rule._SIZE-2 or y==self.rule._SIZE-2:
              ucb = float(ucb) / float(4)
          sys.stderr.write("rate_value:"+str(rate_value))
          #rate_value:0.419258829587
        else:
          ucb=child.rate*4 + C * math.sqrt(math.log(node.child_games_sum)/child.games)

      if ucb > max_ucb:
        max_ucb=ucb
        best_child=child
    if best_child.next_node == self.NoneChild:
      exit(0)
    return best_child,best_child.move

  def _montecarlo(self, node, go_state_obj, player,myself,FIRST_NODE):  # search uct
    (best_child_node, best_child_move) = self.get_best_ucb(node,FIRST_NODE)
    if FIRST_NODE == True:
        FIRST_NODE=False

    #second node in search tree prevent from search using deeplearning

    # Don't modify　copy.deepcopy(). If you delete it,the bug must be occur.
    best_state = self.rule.move_and_return_state(copy.deepcopy(go_state_obj), player, best_child_move)  # TODO:move_and_return_stateはモンテカルロ用のmove関数？

    if best_child_node.games <= 0:  # 最良の子ノードのゲーム数が0の時
      #win = 1 or 0          (-(-1)=0 -(0)=0)
      win = self.playout(go_state_obj, player.next_player,player,myself)  # TODO:プレイアウトで勝率を求める.マイナスになってるのは相手のターンだから？
      init_rate_playout=best_child_node.rate
      best_child_node.rate = float(best_child_node.rate * best_child_node.games + float(0)) / float(best_child_node.games + float(1))
      best_child_node.games = best_child_node.games + 1  # TODO: best_child_node.gmes += 1のほうが良いかも
      node.child_games_sum = node.child_games_sum + 1
      if win == -100:
          return -100
      best_child_node.rate = float(init_rate_playout * best_child_node.games + float(win)) / float(best_child_node.games + float(1))
      #best_child_node.games = best_child_node.games + 1  # TODO: best_child_node.gmes += 1のほうが良いかも
      #node.child_games_sum = node.child_games_sum + 1
      if win == 1:
        return 0
      else:
        return 1
    else:
      # win = -1 or 0 (-(1)=-1) or 1(there is bug)
      #create_nodeは排他制御できる。
      init_rate=best_child_node.rate
      best_child_node.rate = float(best_child_node.rate * best_child_node.games + float(0)) / float(best_child_node.games + float(1))
      best_child_node.games = best_child_node.games + 1  # TODO: best_child_node.gmes += 1のほうが良いかも
      node.child_games_sum = node.child_games_sum + 1

      if best_child_node.next_node == self._EMPTY:  # 最良の子ノードの子ノードが空の時
        best_child_node.next_node = self.create_node(best_state, player)  # ノードの作成

      win = self._montecarlo(best_child_node.next_node, best_state, player.next_player,myself,FIRST_NODE=False)
      if win == -100:
          return -100
      if win == 0:
        #print "win num == 0"
        win_num = 0
      elif win == 1:
        #print "win num == 1"
        win_num =1
      elif win == -1:
        sys.stderr.write("win_num == -1")
        sys.stderr.write("Error!")
        exit(0)
      #best_child_node.rate = float(init_rate * best_child_node.games + float(win_num)) / float(best_child_node.games + float(1))
      #best_child_node.games = best_child_node.games + 1  # TODO: best_child_node.gmes += 1のほうが良いかも
      node.child_games_sum = node.child_games_sum + 1

      if win == 1:
        return 0
      else:
        return 1
  def worker(self,thread_id, go_state_obj,player):
    myself = threading.current_thread()
    while myself.is_running:
      self._montecarlo(self.node,go_state_obj,player,myself,FIRST_NODE=True)

  def thread_close(self,threads):
      #threads=self.threads
      # stop threads
      for thread in threads:
        thread.is_running = False  # hack
      timeout = 3  # seconds
      for thread in threads:
        thread.join(timeout)
        print thread  # for debugging

      # final result
      sys.stderr.write('=== %d threads active ===' % threading.active_count())
      #print 'largest is %.2f' % max(scoreboard)
  def master(self,num_threads, duration,go_state_obj,player):
      stop_time = time.time() + duration
      threads=[]
      for i in range(num_threads):
        thread = threading.Thread(target=self.worker,args=(i, go_state_obj, player))
        thread.is_running = True  # hack
        threads.append(thread)
        thread.start()

      sys.stderr.write('=== %d threads active ===' % threading.active_count())

      # monitor progress of threads
      while time.time() < stop_time:
        time.sleep(1)
      #threads=self.threads
      # stop threads

      for thread in threads:
        thread.is_running = False  # hack
      timeout = 3  # seconds
      for thread in threads:
        thread.join(timeout)
        print thread  # for debugging

      # final result
      sys.stderr.write('=== %d threads finally active ===' % threading.active_count())
      #self.thread_close(threads)
  def next_move(self, go_state_obj, player,left_time,last_opponent_move):#get best uct
    #print self.create_node(player,state._board).child[0]
    self._player = player
    self._ai_player = player

    max_num = -9999999
    best_child = self.NoneChild
    self.first_node_prediction_value=self.first_node_prediction_obj.next_value(self.forward_large_network, self.sess, go_state_obj, player)
    self.node =self.create_node(go_state_obj,player)

    sys.stderr.write("TURNS_NUM:"+str(go_state_obj.turns_num))
    sys.stderr.write(str(left_time))
    time_onemove=float(left_time / float(120+max(120-go_state_obj.turns_num,0)))
    sys.stderr.write("time_onemove:"+str(time_onemove))
    #TODO:modify
    self.master(8,time_onemove/1.8,go_state_obj,player)

    for child in self.node.child:
        sys.stderr.write("games:"+str(child.games))
        sys.stderr.write("rates:"+str(child.rate))
        if child.move != self.rule._PASS:
              #rate:0~1 deep:0~4
              if child.rate*4 + 100*self.first_node_prediction_value[self.xy_to_flat_index(child.move)] > max_num:
                #TODO: rate + DeepLearning Value でmaxとる
                max_num = child.rate*4 + 100*self.first_node_prediction_value[self.xy_to_flat_index(child.move)]
                best_child = child
        else:
              NUM=float(1)
              if go_state_obj.turns_num > 300:
                  NUM = float(2.5)
              if child.rate * NUM > max_num:
                #TODO: rate + DeepLearning Value でmaxとる
                max_num = child.rate+NUM
                best_child = child

    #print state.print_board()
    sys.stderr.write(str("best_child.move============================"))
    sys.stderr.write(str(best_child.move))

    if last_opponent_move == self.rule._PASS:
        sys.stderr.write("PASS! PASS! PASS! opponent passes")
        rnd=random.random()
        if rnd >= 0.5:
            best =self.rule._PASS
    return (self.rule.move(go_state_obj,player,best_child.move), best_child.move)

class DeepLearningMontecarlo(GoVariable):
  def __init__(self):
    self._player = None
    self._ai_player = None
    self._uct_loop=1#The number of trying uct search
    self.NoneChild = Child(None,None)
    self.NoneMove = (-1,-1)
    self.NoneState = Go()
    self.rule=Go()
    from forward_prop_network import ForwardPropNetwork
    self.search_algorithm=DeepLearningSearch()
    self.sess=tf.Session()
    self.forward_prop_network=ForwardPropNetwork(self.sess)

  def playout(self,init_state,player):
    init_player=player
    go_state_obj=copy.deepcopy(init_state)
    board_size=self._SIZE
    roop_num=board_size*board_size+50
    #ok_move_list=[]
    before_move=self.NoneMove
    for i in xrange(roop_num):
        #next_states = [(state.move_and_return_state(copy.deepcopy(state._board),player, move), move)
        #            for move in state.next_moves(player)]
        (go_state_obj, move) = self.search_algorithm.next_move(self.forward_prop_network, self.sess, go_state_obj, player)

        if move==self.rule._PASS and before_move == self.rule._PASS:
            break

        #sys.stderr.write(str(self.rule.print_board(go_state_obj)))
        player = player.next_player
        before_move=move

    score=self.rule.count_territory(go_state_obj)

    score=score[0]-score[1]#black-white

    #If AI player wins,return 1.If Ai lose,return 0.
    if init_player.player_id == self._BLACK:
        if score>=0:
            return 0
        else:
            return -1
    elif init_player.player_id == self._WHITE:
        if score>=0 :
            return -1
        else:
            return 0
  def create_node(self,go_state_obj,player):
    next_moves = self.rule.next_moves(go_state_obj,player)

    node = Node(go_state_obj)
    for move in next_moves:
      #print move
      node.child.append(Child(move,player))
    node.child.append(Child(self._PASS,player))

    node.go_state_obj = go_state_obj

    return node

  def get_best_ucb(self,node):#node=-1 node.next error
    max_ucb=-999999
    best_child = self.NoneChild

    for child in node.child:
      if child.games == 0:
        ucb=1000000
      else:
        C=0.3
        ucb=child.rate + C * math.sqrt(math.log(node.child_games_sum)/child.games)

      if ucb > max_ucb:
        max_ucb=ucb
        best_child=child
    if best_child.next_node == self.NoneChild:
      exit(0)
    return best_child,best_child.move

  def _montecarlo(self,node,go_state_obj,player):#search uct

    (best_child_node, best_child_move) = self.get_best_ucb(node)
    #Don't modify　copy.deepcopy(). If you delete it,the bug must be occur.
    best_state = self.rule.move_and_return_state(copy.deepcopy(go_state_obj), player, best_child_move)

    if best_child_node.games <= 0:
      win = -self.playout(go_state_obj,player.next_player)
    else:
      if best_child_node.next_node == self._EMPTY:
        best_child_node.next_node = self.create_node(best_state,player)
      win = -self._montecarlo(best_child_node.next_node,best_state,player.next_player)

    best_child_node.rate = (best_child_node.rate * best_child_node.games + win) / (best_child_node.games + 1)
    best_child_node.games = best_child_node.games+1
    node.child_games_sum=node.child_games_sum+1

    return win

  def next_move(self, go_state_obj, player):#get best uct
    #print self.create_node(player,state._board).child[0]
    self._player = player
    self._ai_player = player
    node =self.create_node(go_state_obj,player)
    max_num = -9999999
    best_child = self.NoneChild

    for i in xrange(self._uct_loop):
      print "loop"
      self._montecarlo(node,go_state_obj,player)

    for child in node.child:
      print "child.games:"+str(child.games)
      print "child.rate"+str(child.rate)+"    "
      if child.rate > max_num:
        max_num = child.rate
        best_child = child

    #print state.print_board()
    print "best_child.move============================"
    print best_child.move
    return (self.rule.move(go_state_obj,player,best_child.move), best_child.move)
    #search_algorithm=DeepLearningSearch()
    #sess=tf.Session()
    #self.forward_prop_network=ForwardPropNetwork(sess)
    #(go_state_obj, move) = self.search_algorithm.next_move(self.forward_prop_network, self.sess, go_state_obj, player)


    '''
    #tensorflowの場合：
    #tensor_board = output_tensor_board.eval(session=sess)
    #flat_index=output_board.argmax(0)
    #numpy_board=output_board.eval(session=sess)#convert tensor to numpy

    #two_dimensional_np_board=output_board.reshape((19,19))#y,xの順番
    #list_board=two_dimensional_np_board.tolist()

    #[[y,x]] = numpy_move.tolist()

    '''

