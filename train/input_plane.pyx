# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 15:34:21 2016

@author: Tomohiro Ueno
できればどこまでファイルを読み込んだかを記録しておいて、オブジェクトを独立させたい
２次元リストの場合は[[float(1) if ~ else ~ for x] for y]の順番でかく。
ちなみに、board[y][x]の順番。
turns_sinceは石が取られた場合、添え字がおかしくなってしまうので、石が存在しない場合は削除。
"""
import pyximport
pyximport.install()
#from game import Player
#from game import State
#from search import MontecarloSearch
from go import Go
from go import GoStateObject
#import tensorflow as tf
#import math
from go import GoVariable
#from collections import Counter
import copy
import sys

reload(sys)
sys.stderr.write(str(sys.setdefaultencoding('utf-8')))
import numpy as np


class MakeInputPlane(GoVariable):
    def __init__(self):
        self.my_color = None
        self.opponent_color = None
        self.rule = Go()
        self.OPPONENT = None
        self.MY_COLOR = None
        self.DIRECTIONS=([1,0],[0,1],[-1,0],[0,-1])

    def make_legal_move(self,go_state_obj,player):
        #next_moves=self.rule.next_moves(go_state_obj,player)
        legal_board=[[float(1) if self.rule.valid_move_public(go_state_obj,player,(x,y)) else float(0) for x in xrange(self.rule._SIZE)] for y in xrange(self.rule._SIZE)]
        #print legal_board
        return legal_board

    def make_my_color_board(self, go_state_obj, player):
        self.MY_COLOR = player.player_id
        player = player.next_player_func()
        self.OPPONENT = player.player_id
        black_board = [[float(1) if go_state_obj._board[y][x] == self.MY_COLOR else float(0) for x in xrange(self.rule._SIZE)] for y in xrange(self.rule._SIZE)]
        return black_board
    def make_opponent_board(self, go_state_obj, player):
        self.OPPONENT = player.player_id
        player = player.next_player_func()
        self.MY_COLOR = player.player_id
        white_board = [[float(1) if go_state_obj._board[y][x] == self.MY_COLOR else float(0) for x in xrange(self.rule._SIZE)] for y in xrange(self.rule._SIZE)]
        return white_board

    def make_turns_since(self, go_state_obj, player):

        return [[float(go_state_obj._turns_since_board[y][x]) / 4 for x in xrange(self.rule._SIZE)] for y in
                xrange(self.rule._SIZE)]

    def make_dame_board_mycolor(self,go_state_obj,player):
        #sys.stderr.write(str(self.rule.print_board(go_state_obj)))
        player_color=self.PLAYER_ID_TO_COLOR[int(player.player_id)]
        return [[float(self.count_liberties_dame((x, y), go_state_obj._board,player_color)*0.1) if go_state_obj._board[y][x] == player_color else 0 for x in range(self.rule._SIZE)] for y in range(self.rule._SIZE)]

    def make_dame_board_opponent(self,go_state_obj,player):
        #sys.stderr.write(str(self.rule.print_board(go_state_obj)))
        player_color=self.PLAYER_ID_TO_COLOR[int(player.next_player_func().player_id)]
        return [[float(self.count_liberties_dame((x, y), go_state_obj._board,player_color)*0.25) if go_state_obj._board[y][x] == player_color else 0 for x in range(self.rule._SIZE)] for y in range(self.rule._SIZE)]


    def count_liberties_dame(self,target_pos, board, player_color):
        result = 0
        searched_list = list()
        (x0, y0) = target_pos
        if board[y0][x0] == self._EMPTY:
            result = 0
        else:
            queue = [(x0, y0)]
            while len(queue) > 0:
                (x0, y0) = queue.pop(0)
                for dx, dy in self.DIRECTIONS:
                    if result >= 8:
                        return 8
                    (x, y) = (x0 + dx, y0 + dy)
                    if self._on_board(x,y) and (x, y) not in searched_list:
                        searched_list.append((x, y))
                        if board[y][x] == self._EMPTY:
                            result += 1
                        elif board[y][x] == player_color:
                            queue.append((x, y))
        #modified code!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        if result == 1:
            result = 30
        elif result == 2:
            result = 20
        elif result >= 10:
            result = 1
        else:
            result = 10 - result + 1

        return result

    def make_captured_map(self, go_state_obj, player):

        self.OPPONENT = player.player_id
        player = player.next_player_func()
        self.MY_COLOR = player.player_id

        #next_move_list = self.rule.next_moves(go_state_obj, player)
        #capture_size_board = [[float(self.count_captures((x, y), copy.deepcopy(go_state_obj._board), player)) if (x,y) in next_move_list else float(0) for x in xrange(self.rule._SIZE)] for y in xrange(self.rule._SIZE)]

        capture_size_board = [[float(self.count_captures((x, y), copy.deepcopy(go_state_obj._board), player)) if self.rule.valid_move_public(go_state_obj,player,(x,y)) else float(0) for x in xrange(self.rule._SIZE)] for y in xrange(self.rule._SIZE)]

        return capture_size_board

    def make_captures_map(self, go_state_obj, player):
        self.MY_COLOR = player.player_id
        self.OPPONENT = player.next_player_func().player_id
        #next_move_list = self.rule.next_moves(go_state_obj, player)

        capture_size_board = [[float(self.count_captures((x, y), copy.deepcopy(go_state_obj._board), player)) if self.rule.valid_move_public(go_state_obj,player,(x,y)) else float(0) for x in xrange(self.rule._SIZE)] for y in xrange(self.rule._SIZE)]

        return capture_size_board

    def count_liberties(self, target_pos, board):
        #囲まれていたら0を返す。Go()クラスの方も直さないといけないかも
        result = 0
        searched_list = list()
        (x0, y0) = target_pos
        if board[y0][x0] != self.OPPONENT:
            result = 0
        else:
            queue = [(x0, y0)]
            while len(queue) > 0:
                (x0, y0) = queue.pop(0)
                for (dx, dy) in self.DIRECTIONS:
                    (x, y) = (x0 + dx, y0 + dy)
                    if self._on_board(x, y) and (x, y) not in searched_list:
                        searched_list.append((x, y))
                        if board[y][x] == self._EMPTY:
                            result += 1
                        elif board[y][x] == self.OPPONENT:
                            queue.append((x, y))
        return result

    def count_captures(self, target_pos, board, player):

        result = 0
        flips = list()
        (x0, y0) = target_pos
        if board[y0][x0] != self.rule._EMPTY:
            return 0
        else:
            board[y0][x0] = self.MY_COLOR
            for (dx, dy) in self.rule._DIRECTIONS:
                (x, y) = (x0 + dx, y0 + dy)
                if not self._on_board(x, y) or board[y][x] == self.MY_COLOR or board[y][
                    x] == self.rule._EMPTY or self.count_liberties((x, y), copy.deepcopy(board)) != 0:
                    continue
                elif board[y][x] == self.OPPONENT:
                    flips.extend(self.count_capture((x, y), copy.deepcopy(board)))

                    result = len(set(flips))
            board[y0][x0] = self.rule._EMPTY

            return result

    def count_capture(self, target_pos, board, searched_list=None):
        myself = self.OPPONENT
        result = list()
        result.append(target_pos)
        (x0, y0) = target_pos
        if searched_list is None:
            searched_list = list()
        if not (x0, y0) in searched_list:
            searched_list.append((x0, y0))
        for (dx, dy) in self.rule._DIRECTIONS:
            (x, y) = (x0 + dx, y0 + dy)
            if not self._on_board(x, y):
                continue
            if board[y][x] == myself and (x, y) not in searched_list:
                result.extend(self.count_capture((x, y), copy.deepcopy(board), searched_list))

        return result

    def _on_board(self, x, y):
        '''
        :param x: 盤面の水平の座標
        :param y: 盤面の垂直の座標
        :return: ボード上にあるかどうか(True or False)
        '''
        if 0 <= x < self.rule._SIZE and 0 <= y < self.rule._SIZE:
            return True
        else:
            return False

    def _check_surrounded(self, go_state_obj, player, next_move):
        '''
        :param player: 囲まれてるか判定する石のプレーヤー
        :param next_move: 次の手の座標(x, y)
        :return: 次の手が囲まれているかどうかを返す(囲まれていたらTrue)
        '''

        searched_list = []
        myself = player.player_id

        queue = [next_move]
        while len(queue) > 0:
            (x0, y0) = queue.pop(0)
            for (dx, dy) in self.rule._DIRECTIONS:
                (x, y) = (x0 + dx, y0 + dy)
                if self._on_board(x, y) and (x, y) not in searched_list:
                    searched_list.append((x, y))
                    if go_state_obj._board[y][x] == self.rule._EMPTY:
                        return False
                    elif go_state_obj._board[y][x] == myself:
                        queue.append((x, y))

        return True

    def generate_input(self, go_state_obj, player):
        self.my_color = player.player_id
        self.opponent_color = player.next_player.player_id

        legal_board = self.make_legal_move(copy.deepcopy(go_state_obj),copy.deepcopy(player))

        black_board = self.make_my_color_board(copy.deepcopy(go_state_obj), copy.deepcopy(player))

        white_board = self.make_opponent_board(copy.deepcopy(go_state_obj), copy.deepcopy(player))

        dame_board_my = self.make_dame_board_mycolor(copy.deepcopy(go_state_obj), copy.deepcopy(player))

        dame_board_opponent = self.make_dame_board_opponent(copy.deepcopy(go_state_obj), copy.deepcopy(player))

        captures_board = self.make_captures_map(copy.deepcopy(go_state_obj), copy.deepcopy(player))

        captured_board = self.make_captured_map(copy.deepcopy(go_state_obj), copy.deepcopy(player))

        turns_since_board = self.make_turns_since(copy.deepcopy(go_state_obj), copy.deepcopy(player))

        flatten_xs = [legal_board,black_board, white_board, dame_board_my,dame_board_opponent, captures_board, captured_board, turns_since_board]
        #print flatten_xs
        return flatten_xs
    def generate_answer(self, pos_tuple):

        #y_board=[float(1) if (x, y) is pos_tuple else float(0) for y in xrange(self.rule._SIZE) for x in xrange(self.rule._SIZE)]
        #flatten_y=np.reshape(y_board,361)
        y_board = [[float(1) if (x, y) == pos_tuple else float(0) for x in xrange(self.rule._SIZE)] for y in xrange(self.rule._SIZE)]

        return y_board
    '''
    def _test(self):
        go_state_obj=GoStateObject()


        ======================
        これじゃうまくいかない
        go_state_obj._board=[[-1,-1,-1,-1,-1,-1,-1,-1,-1],
                             [-1,1,-1,-1,-1,-1,-1,-1,-1],
                             [1,0,1,-1,-1,-1,-1,-1,-1],
                             [-1,-1,-1,-1,-1,-1,-1,-1,-1],
                             [-1,-1,-1,-1,-1,-1,-1,-1,-1],
                             [-1,-1,-1,-1,-1,-1,-1,-1,-1],
                             [-1,-1,-1,-1,-1,-1,-1,-1,-1],
                             [0,0,-1,-1,-1,-1,-1,-1,-1],
                             [1,-1,-1,-1,-1,-1,-1,-1,-1],
                             ]
        =========================
        players = [Player(0, 'human'), Player(1, 'human')]
        players[0].next_player = players[1]
        players[1].next_player = players[0]
        player = players[0]#player's color is 0
        #self.make_capture_size_boards(go_state_obj,player)
        self.my_color=player
        self.opponent_color=player.next()
        =================================
        go_state_obj._board[0][0]=self.rule._BLACK
        go_state_obj._board[0][1]=self.rule._WHITE
        go_state_obj._board[1][0]=self.rule._WHITE
        go_state_obj._board[1][1]=self.rule._BLACK
        go_state_obj._board[2][0]=self.rule._WHITE
        go_state_obj._board[0][2]=self.rule._BLACK
        #go_state_obj._board[2][2]=self.rule._WHITE
        print go_state_obj._board
        =================================
        return self.make_captured_map(go_state_obj,player)
        #return self.make_captures_map(go_state_obj,player)
    '''
