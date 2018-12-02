#coding:utf-8
import sys

# sysモジュールをリロードする
reload(sys)
# デフォルトの文字コードを変更する
sys.stderr.write(str(sys.setdefaultencoding('utf-8')))



class Player(object):
    def __init__(self, player_id, player_name):
        assert player_id in (0, 1)
        self.player_id = player_id
        self.player_name = player_name
        self.next_player = None

    def next_player_func(self):
        return self.next_player

    def __str__(self):
        return self.player_name


class State(object):
    '''Class to store the game state.'''
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def valid_move(self, player, next_move):
        '''Returns True if next_move is a valid move for the player.'''
        raise NotImplementedError

    def next_moves(self, player):
        '''Returns a list of next moves for the player.'''
        raise NotImplementedError

    def move(self, player, next_move):
        '''Returns the new state when the player makes next_move.'''
        raise NotImplementedError

    def win(self, player):
        '''Returns true if the player wins.'''
        raise NotImplementedError

    def draw(self):
        '''Returns true if the game draws.'''
        raise NotImplementedError

    def score(self, player):
        '''Returns a real or an estimated score for the player.
        If the computation is time-consuming, consider caching the result.'''
        raise NotImplementedError

    def serialize(self):
        '''Returns a hashable object representing the state.
        Use tuple instead of list, and frozenset instead of set.'''
        raise NotImplementedError
