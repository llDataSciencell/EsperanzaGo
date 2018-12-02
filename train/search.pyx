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
  def next_move(self,train,sess,go_state_obj, player):
    output_board=train.search_deep_learning(sess,go_state_obj,player)

    list_board=output_board.tolist()
    #sys.stderr.write(str(list_board))
    next_move_list=self.rule.next_moves_flat(go_state_obj,player,True)
    #sys.stderr.write(str(next_move_list))

    for y in xrange(self._SIZE):
      for x in xrange(self._SIZE):
        if ([y*self._SIZE+x] not in next_move_list):
          list_board[0][y*self._SIZE+x]=None

    flat_move=list_board[0].index(max(list_board[0]))

    (x,y)=self.two_dimensional_index(flat_move)

    if self.rule.valid_move_public(go_state_obj,player,(x,y)):
      next_move_pos = (x,y)
    else:
      print "passssssspasssssspasssss"
      next_move_pos=self.rule._PASS

    return (self.rule.move_and_return_state(go_state_obj, player, next_move_pos), next_move_pos)


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
    self._uct_loop=50#The number of trying uct search
    self.NoneChild = Child(None,None)
    self.NoneMove = (-1,-1)
    self.NoneState = Go()
    self.rule=Go()



  def clockwise(self,p):
      q = [[p[2][0],p[1][0],p[0][0]],
           [p[2][1],p[1][1],p[0][1]],
           [p[2][2],p[1][2],p[0][2]] ]

      return q


  def reverse(self,p):
      q = [[p[0][2],p[0][1],p[0][0]],
           [p[1][2],p[1][1],p[1][0]],
           [p[2][2],p[2][1],p[2][0]] ]

      return q

  letter_kind = GoStateObject()
  _=letter_kind._EMPTY
  x=letter_kind._BLACK
  o=letter_kind._WHITE
  B=float(2)#state is outside of board
  A=float(3)#any kind of state is ok
  w_pattern = [[[x, o, o],
                [_, _, _],
                [A, A, A]]]

  b_pattern = [[[o, o, x],
                [x, _, x],
                [_, o, _]]]
  def make_pattern(self):
    pass

  def pattern_match(self,board,player):

    BOARD_SIZE = 19
    PATTERN_NUM = 1
    BLOCK_SIZE = 3

    if(player == 0):
        pattern = self.w_pattern
    else:
        pattern = self.b_pattern

    match_point = []
    pattern_match_flag = 0

    for i in range(PATTERN_NUM):
        expand_pattern = self.get_expand_pattern(pattern[i])

        for j in range(self.rule._SIZE-BLOCK_SIZE):
            for k in range(self.rule._SIZE-BLOCK_SIZE):
                ### get block ###
                block = [[board[j][k]  ,board[j+1][k]  ,board[j+2][k]  ],
                         [board[j][k+1],board[j+1][k+1],board[j+2][k+1]],
                         [board[j][k+2],board[j+1][k+2],board[j+2][k+2]]]

                ### pattern matching ###
                for l in range(8):
                    match_flag = 0
                    for m in range(BLOCK_SIZE):
                        for n in range(BLOCK_SIZE):
                            if(block[m][n] == expand_pattern[l][m][n]):
                                match_flag += 1

                    if(match_flag == BLOCK_SIZE ** 2):
                        match_point.append([j,k])
                        pattarn_match_flag = 1
        if(pattern_match_flag == 1):
            break;

    #show match_point: [[0~18,0~18],...]
    print("match_point: {0}").format(match_point)


  ### expand patterns ###
  #p:pattern
  def get_expand_pattern(self,p):
          expand_pattern = []
          expand_pattern.append(p)
          expand_pattern.append(self.clockwise(expand_pattern[0]))
          expand_pattern.append(self.clockwise(expand_pattern[1]))
          expand_pattern.append(self.clockwise(expand_pattern[2]))
          expand_pattern.append(self.reverse(expand_pattern[3]))
          expand_pattern.append(self.clockwise(expand_pattern[4]))
          expand_pattern.append(self.clockwise(expand_pattern[5]))
          expand_pattern.append(self.clockwise(expand_pattern[6]))

          return expand_pattern

  def playout(self,init_state,player):
    init_player=player
    go_state_obj=copy.deepcopy(init_state)
    board_size=self.rule.get_board_size()
    roop_num=board_size*board_size+50
    #ok_move_list=[]
    before_move=self.NoneMove
    previous_place=self.rule._PASS
    for i in xrange(roop_num):
        #next_states = [(state.move_and_return_state(copy.deepcopy(state._board),player, move), move)
        #            for move in state.next_moves(player)]
        if before_move != self.rule._PASS:
          (before_x,before_y) =before_move

        empty_list=[]
        for x in xrange(self._SIZE):
          for y in xrange(self._SIZE):
            if go_state_obj._board[y][x] != self.rule._EMPTY:
              continue
            if before_move != self.rule._PASS:
              if abs(x-before_x) == 1 or abs(y-before_y)==1:
                self.pattern_match(go_state_obj._board,player)

        next_moves = self.rule.next_moves(go_state_obj,player, True)

        move=random.choice(next_moves)

        if move=="pass" and before_move == "pass":
            break;

        go_state_obj=self.rule.move(go_state_obj,player,move)
        print self.rule.print_board(go_state_obj)
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
    print next_moves
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
    max = -9999999
    best_child = self.NoneChild

    for i in xrange(self._uct_loop):
      print "loop"
      self._montecarlo(node,go_state_obj,player)

    for child in node.child:
      print "child.games -> child.rate"
      print child.games
      print child.rate
      if child.rate > max:
        max = child.rate
        best_child = child

    #print state.print_board()
    print "best_child.move============================"
    print best_child.move
    return (self.rule.move(go_state_obj,player,best_child.move), best_child.move)

class DeepLearningMontecarlo(GoVariable):
  def __init__(self):
    self._player = None
    self._ai_player = None
    self._uct_loop=50#The number of trying uct search
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

        if move=="pass" and before_move == "pass":
            break

        print self.rule.print_board(go_state_obj)
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
    print next_moves
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
    max = -9999999
    best_child = self.NoneChild

    for i in xrange(self._uct_loop):
      print "loop"
      self._montecarlo(node,go_state_obj,player)

    for child in node.child:
      print "child.games -> child.rate"
      print child.games
      print child.rate
      if child.rate > max:
        max = child.rate
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

