#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:18:32 2017

@author: user
"""
import pyximport; pyximport.install()
from input_plane import MakeInputPlane
import sys
import unittest
from go import GoStateObject
from go import Go
from game import Player
from game import State
from search import MontecarloSearch
from go import Go
from go import GoStateObject
import tensorflow as tf
import math
from go import GoVariable
import copy
import sys
reload(sys)
sys.stderr.write(str(sys.setdefaultencoding('utf-8')))
class StateSettingBlack(object):
    def __init__(self):
        self.go_state_obj = GoStateObject()
        self.rule = Go()
        self.players = [Player(0, 'human'), Player(1, 'human')]
        self.players[0].next_player = self.players[1]
        self.players[1].next_player = self.players[0]
        self.player = self.players[0]  # player's color is 0
        # self.make_capture_size_boards(go_state_obj,player)
        self.plane = MakeInputPlane()
        self.plane.my_color = self.player
        self.plane.opponent_color = self.player.next_player_func()
        print "state_setting obj is created"
class StateSettingWhite(object):
    def __init__(self):
        self.go_state_obj = GoStateObject()
        self.rule = Go()
        self.players = [Player(0, 'human'), Player(1, 'human')]
        self.players[0].next_player = self.players[1]
        self.players[1].next_player = self.players[0]
        self.player = self.players[1]  # player's color is 1
        # self.make_capture_size_boards(go_state_obj,player)
        self.plane = MakeInputPlane()
        self.plane.my_color = self.player
        self.plane.opponent_color = self.player.next_player_func()
        print "state_setting obj is created"

class RuleTest(unittest.TestCase,StateSettingBlack,StateSettingWhite):
        def test_one_caputure_black_player(self):
            state=StateSettingBlack()
            rule=Go()
            #state.go_state_obj.set_board_size(5)
            state.go_state_obj._board=rule.translate_char_to_int([
            ".ox..",
            ".xo..",
            ".ox..",
            ".....",
            ".....",])
            #空白のボードに石を置く
            #print state.plane.make_captures_map(copy.deepcopy(state.go_state_obj), state.player)
            #print state.plane.count_captures((3,1),copy.deepcopy(state.go_state_obj._board), state.player)
            self.assertEqual(state.plane.make_captures_map(copy.deepcopy(state.go_state_obj),state.player),
                [[1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]
            )
        def test_one_caputure_white_player(self):
            state=StateSettingWhite()
            rule=Go()
            #state.go_state_obj.set_board_size(5)
            state.go_state_obj._board=rule.translate_char_to_int([
            ".xo..",
            ".ox..",
            ".xo..",
            ".....",
            ".....",])
            #空白のボードに石を置く
            #print state.plane.make_captures_map(copy.deepcopy(state.go_state_obj), state.player)
            #print state.plane.count_captures((3,1),copy.deepcopy(state.go_state_obj._board), state.player)
            self.assertEqual(state.plane.make_captures_map(copy.deepcopy(state.go_state_obj),state.player),
                [[1,0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]
            )

        def test_blank(self):
            state=StateSettingBlack()
            rule=Go()
            #state.go_state_obj.set_board_size(5)
            state.go_state_obj._board=rule.translate_char_to_int([
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",])
            #print state.go_state_obj
            #空白のボードに石を置く
            self.assertEqual(state.plane.make_captured_map(state.go_state_obj,state.player),
            [[0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]]
            )


        def test_three_caputures_white_player(self):
            state=StateSettingWhite()
            rule=Go()
            #state.go_state_obj.set_board_size(5)
            state.go_state_obj._board=rule.translate_char_to_int([
            "xo...",
            ".xo..",
            "xo...",
            "o....",
            ".....",])
            #空白のボードに石を置く
            self.assertEqual(state.plane.make_captures_map(state.go_state_obj,state.player),
                [[0, 0, 0, 0, 0],
                [3, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]
            )

        def test_many_capure_white(self):
            state=StateSettingWhite()
            rule=Go()
            #state.go_state_obj.set_board_size(5)
            state.go_state_obj._board=rule.translate_char_to_int([
            "xoo..",
            ".xxo.",
            "xxo..",
            "oo...",
            ".....",])
            #空白のボードに石を置く
            self.assertEqual(state.plane.make_captures_map(state.go_state_obj,state.player),
                [[0, 0, 0, 0, 0],
                [5, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]
            )
        def test_my_color(self):
            state=StateSettingWhite()
            rule=Go()
            #state.go_state_obj.set_board_size(5)
            state.go_state_obj._board=rule.translate_char_to_int([
            "xooo.",
            "xooo.",
            "xooo.",
            ".....",
            ".....",])
            #空白のボードに石を置く
            self.assertEqual(state.plane.make_my_color_board(state.go_state_obj,state.player),
                [[0,1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]
            )
        def test_opponent_color(self):
            state=StateSettingBlack()
            rule=Go()
            #state.go_state_obj.set_board_size(5)
            state.go_state_obj._board=rule.translate_char_to_int([
            "xooo.",
            "xooo.",
            "xooo.",
            ".....",
            ".....",])
            #空白のボードに石を置く
            self.assertEqual(state.plane.make_my_color_board(state.go_state_obj,state.player),
                [[1,0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]
            )

        def test_turns_since(self):
            state=StateSettingWhite()
            rule=Go()
            #state.go_state_obj.set_board_size(5)

            state.go_state_obj=rule.move(state.go_state_obj,state.player,(2,2))
            rule.move(state.go_state_obj,state.player,(2,2))
            '''
            state.go_state_obj._board=rule.translate_char_to_int([
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",])

            state.go_state_obj._turns_since_board = \
                [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]
            '''
            #空白のボードに石を置く
            self.assertEqual(state.plane.make_turns_since(state.go_state_obj,state.player),
                [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]
            )

        def test_turns_sinces(self):
            state=StateSettingWhite()

            rule=Go()
            #state.go_state_obj.set_board_size(5)

            state.go_state_obj=rule.move(state.go_state_obj,state.player,(2,2))
            rule.move(state.go_state_obj,state.player,(2,2))
            state.go_state_obj = rule.move(state.go_state_obj, state.player.next_player_func(), (1, 1))
            rule.move(state.go_state_obj,state.player,(1,1))
            '''
            state.go_state_obj._board=rule.translate_char_to_int([
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",])

            state.go_state_obj._turns_since_board = \
                [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]
            '''
            #空白のボードに石を置く
            self.assertEqual(state.plane.make_turns_since(state.go_state_obj,state.player),
                [[0, 0, 0, 0, 0],
                [0, 2.0, 0,0, 0],
                [0, 0,1.75,0, 0],
                [0, 0, 0, 0,  0],
                [0, 0, 0, 0,  0]]
            )

        def test_turns_sinces_over8(self):
            state=StateSettingWhite()

            rule=Go()
            #state.go_state_obj.set_board_size(5)

            state.go_state_obj=rule.move(state.go_state_obj,state.player,(0,0))
            rule.move(state.go_state_obj,state.player,(0,0))
            #print state.go_state_obj._turns_since_board
            state.go_state_obj = rule.move(state.go_state_obj, state.player.next_player_func(), (0, 4))
            rule.move(state.go_state_obj,state.player,(0,4))
            #print state.go_state_obj._turns_since_board
            state.go_state_obj=rule.move(state.go_state_obj,state.player,(1,0))
            rule.move(state.go_state_obj,state.player,(1,0))
            #print state.go_state_obj._turns_since_board
            state.go_state_obj = rule.move(state.go_state_obj, state.player.next_player_func(), (1, 4))
            rule.move(state.go_state_obj,state.player,(1,4))
            #print state.go_state_obj._turns_since_board
            state.go_state_obj=rule.move(state.go_state_obj,state.player,(2,0))
            rule.move(state.go_state_obj,state.player,(2,0))
            #print state.go_state_obj._turns_since_board
            state.go_state_obj = rule.move(state.go_state_obj, state.player.next_player_func(), (2, 4))
            rule.move(state.go_state_obj,state.player,(2,4))
            #print state.go_state_obj._turns_since_board
            state.go_state_obj=rule.move(state.go_state_obj,state.player,(3,0))
            rule.move(state.go_state_obj,state.player,(3,0))
            #print state.go_state_obj._turns_since_board
            state.go_state_obj = rule.move(state.go_state_obj, state.player.next_player_func(), (3, 4))
            rule.move(state.go_state_obj,state.player,(3,4))
            #print state.go_state_obj._turns_since_board
            state.go_state_obj=rule.move(state.go_state_obj,state.player,(4,0))
            rule.move(state.go_state_obj,state.player,(4,0))
            #print state.go_state_obj._turns_since_board
            state.go_state_obj = rule.move(state.go_state_obj, state.player.next_player_func(), (4, 4))
            rule.move(state.go_state_obj,state.player,(4,4))
            #print state.go_state_obj._turns_since_board
            #空白のボードに石を置く
            self.assertEqual(state.plane.make_turns_since(state.go_state_obj,state.player),
                [[0, 0.25, 0.75, 1.25, 1.75],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0.5, 1.0, 1.5, 2.0]]
            )
        '''
        def test_capture_move(self):
            state=StateSettingWhite()
            rule=Go()
            state.go_state_obj.set_board_size(5)
            state.go_state_obj._board=rule.translate_char_to_int([
            "xoo..",
            ".xxo.",
            "xxo..",
            "oo...",
            ".....",])
            #空白のボードに石を置く
            print state.rule.print_board(state.rule.move(state.go_state_obj, state.player, (1, 2)))
            self.assertEqual(state.rule.move(state.go_state_obj,state.player,(0,1))._board,
            rule.translate_char_to_int(
            [
            ".oo..",
            "o..o.",
            "..o..",
            "oo...",
            ".....",]
            ))
        '''
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(RuleTest)
    unittest.TextTestRunner(verbosity=2, stream=sys.stderr).run(suite)
        
        