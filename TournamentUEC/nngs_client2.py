#coding:utf-8
# step1 open a Terminal
# step2 python nngs_player1.py
# step3 open another Terminal
# step4 python nngs_player2.py player1 B
from __future__ import print_function
import pyximport
pyximport.install()
from go import GoStateObject
from search import MontecarloSearch
import sys
import socket
import numpy as np
from contextlib import closing

import random

from game import Player
from go import Go


class NNGS:
  STONE_COLOR = {"B":0, "W":1}
  HOST = 'jsb.cs.uec.ac.jp'
  PORT = 9696
  BUFSIZE = 2048
  BOARD_COLUMS = "ABCDEFGHJKLMNOPQRST"
  OPPONENT_COLOR = {"B":"W", "W":"B"}

  def __init__(self):
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.str_data = None
    self.player_name = {"B":None, "W":None}
    self.user_name = None
    self.my_color = None
    self.opponent_name = None
    self.rule=Go()
    self.time_data={"B":None,"W":None}

  def send_data(self, s):
    '''send data to nngs server'''
    if s[-1] != "\n":
      s = s + "\n"
    self.sock.send(s.encode('utf-8'))

  def receive_data(self):
    '''receive data from nngs server'''
    self.str_data = self.sock.recv(NNGS.BUFSIZE)
    print(self.str_data, end='')
    return len(self.str_data)

  def connect_nngs(self):
    '''connect nngs server'''
    self.sock.connect((NNGS.HOST, NNGS.PORT))

  def close_nngs(self):
    '''close nngs server'''
    closing(self.sock)
    print("closed")

  def login(self, user_name="espgo"):
    '''login'''
    self.user_name = user_name
    self.send_data(self.user_name+"\n")
    self.receive_data()

  def match(self, opponent_name, my_color):
    '''match'''
    self.my_color = my_color
    self.opponent_name = opponent_name
    self.player_name[self.my_color] = self.user_name
    self.player_name[NNGS.OPPONENT_COLOR[self.my_color]] = self.opponent_name
    if self.player_name["B"] is not None and self.player_name["W"] is not None:
      self.send_data("match %s %s 19 30 0\n" % (self.opponent_name, self.my_color))

  def time(self):
    '''request time'''
    # TODO: test
    #print("[time run!]")
    self.send_data("time")
    return self.nngs_wait()

  def accept(self):
    '''when requested, accept the match'''
    if "<match" in self.str_data and "<decline" in self.str_data:
      first = self.str_data.find("<match") + 1
      end = self.str_data.find("> or ")
      tmp = self.str_data[first:end].split(" ")
      self.match(tmp[1], tmp[2])

  def nngs_wait(self):
    '''wait match and opponent next move'''
    while True:
      if self.receive_data() <= 0: return -1
      # simple mode
      if "No Name Go Server" in self.str_data: self.send_data("set client TRUE\n")
      # match requested
      if "Match [19x19]" in self.str_data:
        print("get a match")
        self.accept()
      # match accepted
      if "accepted." in self.str_data:
        return -1
      # error
      if "Illegal" in self.str_data: return -1
      # get move
      if self.my_color is not None and self.opponent_name is not None:
        tmp = "(%s): " % NNGS.OPPONENT_COLOR[self.my_color]
        fp = self.str_data.find(tmp)
        if fp > 0:
          ep = self.str_data.find("\r", fp)
          move = self.str_data[fp+5:ep]
          if "Pass" in move:
            print("opponent pass")
            return 0
          else:
            x, y = NNGS.BOARD_COLUMS.find(move[0])+1, int(move[1:])
            return y * 21 + x
      if self.my_color is not None and self.opponent_name is not None:
        white_time = "White(%s)" % self.player_name['W']
        if white_time in self.str_data:
            lines = self.str_data.split('\n')
            self.time_data['W'] = map(int, lines[1][lines[1].find(':') + 2:lines[1].find('\r')].split(':'))
            self.time_data['B'] = map(int, lines[2][lines[2].find(':') + 2:lines[2].find('\r')].split(':'))
            return self.time_data
      # pass x 2
      if "You can check your score" in self.str_data:
        self.send_data("done\n")
        continue;
      # "9 {Game 1: test2 vs test1 : Black resigns. W 10.5 B 6.0}"
      if "9 {Game " in self.str_data and "resigns." in self.str_data:
        tmp = "%s vs %s" % (self.player_name["W"], self.player_name["B"])
        if tmp in self.str_data: return -1
      # connect closed case
      if self.my_color is not None and self.opponent_name is not None:
        tmp = "{%s has disconnected}" % self.player_name[NNGS.OPPONENT_COLOR[self.my_color]]
        if tmp in self.str_data: return -1
      # has adjourned
      if "has adjourned." in self.str_data: return -1
      # time out case
      if "forfeits on time" in self.str_data and \
       self.player_name["B"] in self.str_data and self.player_name["W"]: return -1


def move2nngs(move,rule):
  '''invert move(tuple) to nngs(string)'''
  # print("move2nngs:", move)
  if move == rule._PASS:
    next_move = rule._PASS
  else:
    (x, y) = move
    next_move = "%s%d\n" % (NNGS.BOARD_COLUMS[(x+1)-1], y+1)
  return next_move


def nngs2move(state, player, z):
  '''invert z(int) to move(tuple or int)'''
  # TODO:modify move Type
  if z == 0:
    return 3
  else:
    y = z / 21
    x =  z - y * 21
    print("nngs2move:(%d, %d)" % (x, y))
    if state.valid_move(player, (x-1, y-1)):
      return (x-1, y-1)
    else:
      print("[nngs2move] valid :", (x, y))
      return (x-1,y-1)

# ex) python nngs_player1.py
def main():
  #TODO delete code from here
  go_state_obj=GoStateObject()
  from search import DeepLearningSearch
  from forward_prop_network import ForwardPropNetwork
  import tensorflow as tf
  import time
  #search_algorithm = DeepLearningSearch()
  sess = tf.Session()
  #forward_prop_network = ForwardPropNetwork(sess)
  import copy

  search_algorithm=MontecarloSearch()
  go_state_obj=GoStateObject()

  args = sys.argv
  user = 'esp2'


  # NNGS set up
  nngs = NNGS()
  # connect nngs
  nngs.connect_nngs()
  # login
  nngs.login(user)
  # match
  if len(args) == 3 and args[2] in ('B','W'):
    nngs.match(args[1], args[2])
  # wait
  nngs.nngs_wait()

  init_time=time.time()
  sum_time=0
  # Go, Player set up
  rule = Go()
  players = [Player(0, nngs.player_name['B']), Player(1, nngs.player_name['W'])]
  players[0].next_player = players[1]
  players[1].next_player = players[0]
  player = players[0]
  #last_opponent_move=None
  last_opponent_move = None
  while True:
    print(rule.print_board(go_state_obj))
    if player.player_name == user:
      my_start_time=time.time()
      (go_state_obj, move) = search_algorithm.next_move(go_state_obj, player,1800-(float(sum_time)),last_opponent_move)
      print("next_move:", move)
      #go_state_obj = rule.move_and_return_state(go_state_obj, player, move)
      nngs.send_data(move2nngs(move,rule))
      my_end_time=time.time()
      sum_time=sum_time+my_end_time-my_start_time
      print("Sum time"+str(sum_time))
      #search_algorithm.thread_close()
    else:
      z = nngs.nngs_wait()
      if z == None:
        print("None! None! なん！")
        exit(0)
      if z < 0: continue
      #go_state_obj = rule.move_and_return_state(go_state_obj,player, nngs2move(rule, player, z))
      nn_move = nngs2move(rule, player, z)
      print(nn_move)
      if nn_move == 0 or nn_move ==3:
        last_opponent_move=rule._PASS
      else:
        last_opponent_move = nn_move
        go_state_obj = rule.move_and_return_state(go_state_obj, player, nn_move)


    player = player.next_player

  nngs.close_nngs()


if __name__ == '__main__':
  main()
