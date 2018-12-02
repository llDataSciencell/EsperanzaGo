# coding:utf-8
# python2.7
# ボードの状態とルールの実装部分を切り離しました。
# 注意事項：move,valid_move,move_and_return_boardなどは一旦Goオブジェクトに碁盤の情報を代入してから計算する。
# moveの返却値はGoStateObjectのオブジェクトのみとなっており、メモリの節約をしている。
import copy
from game import State
import sys
from game import Player
reload(sys)
sys.stderr.write(str(sys.setdefaultencoding('utf-8')))  # cythonの変換のためらしい


class GoVariable(object):
    # EMPTY:-1 BLACK:0 WHITE:1
    _SIZE = 19# sgf_modeの時は19に設定
    _EMPTY = float(-1.0)
    _PASS = "pass"

    _BLACK = float(0)
    _WHITE = float(1)
    PLAYER_ID_TO_COLOR = [_BLACK, _WHITE]

    def __init__(self):
        pass


class GoStateObject(GoVariable):
    def __init__(self, board=None, ko_place=None, turns_since_board=None, turns_num=None):
        if board is not None:
            self._board = board
            self._ko_place = ko_place
            self._turns_since_board = turns_since_board
            self.turns_num = turns_num
        else:
            self._board = [[float(self._EMPTY)] * self._SIZE for _ in xrange(self._SIZE)]
            self._ko_place = []
            self._turns_since_board = [[float(0) for x in xrange(self._SIZE)] for _ in xrange(self._SIZE)]
            self.turns_num = 0


class Go(State, GoVariable):
    _DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))  # 四方を探索するための変数
    _BT = 0.5  # Black Territory
    _WT = 1.5  # White Territory

    black_player = Player(0, 'human')
    white_player = Player(1, 'human')
    black_player.next_player = white_player
    white_player.next_player = black_player


    #TODO： __init__直す！

    def __init__(self, board=None, ko_place=None):
        if board is not None:
            self._board = board
            self._ko_place = ko_place
        else:
            self._board = [[float(self._EMPTY)] * self._SIZE for _ in xrange(self._SIZE)]
            self._ko_place = []

    def translate_char_to_int(self, charBoard):
        intBoard = []

        for ls in charBoard:
            branchList = []
            for c in ls:
                if c == u".":
                    branchList.extend([self._EMPTY])
                elif c == u"x":
                    branchList.extend([self._BLACK])
                elif c == u"o":
                    branchList.extend([self._WHITE])
                else:
                    sys.stderr.write(str("Board Error! The character of the stone is not defined."))
                    return
            intBoard.append(branchList)

        sys.stderr.write(str(intBoard))
        return intBoard

    def translate_int_to_char(self, intBoard):
        charBoard = []

        for ls in intBoard:
            branchList = []
            for c in ls:
                if c == self._EMPTY:
                    branchList.extend(".")
                elif c == self._BLACK:
                    branchList.extend("x")
                elif c == self._WHITE:
                    branchList.extend("o")
                else:
                    sys.stderr.write(str("Board Error! The character of the stone is not defined."))
                    return
            charBoard.append(branchList)

        charBoard = [''.join(x) for x in charBoard]
        #sys.stderr.write(str( charBoard
        return charBoard

    def create_territory_map(self, board):
        # 改正した（塚原）
        territoryBoard = copy.deepcopy(board)
        # print "[mt]territoryTable:" + str(territoryBoard)
        for x in range(self._SIZE):
            for y in range(self._SIZE):
                if territoryBoard[x][y] == self._EMPTY:  # 空白は陣地として数える
                    point = (x, y)
                    territoryBoard = self.checkTerritory(point, territoryBoard)
                    #TODO:delete break

        return territoryBoard

    def checkTerritory(self, point, territoryBoard):
        # 改正した（塚原）
        x, y = point
        around_list = list()

        # 残りを走査　空白ならcontinue 石を見つけたら、その色をneightboarhoodに追加。

        for j in xrange(1,self._SIZE-y):
            around, flag = self.add_state_to_around(territoryBoard[x][y + j])[:]
            if around != None:
                around_list.append(around)
            if flag:
                break

        for i in xrange(1,self._SIZE-x):
            around, flag = self.add_state_to_around(territoryBoard[x + i][y])[:]
            if around != None:
                around_list.append(around)
            if flag:
                break

        for i, j in zip(range(1, self._SIZE), range(1, self._SIZE)):
            if x + i < self._SIZE and y - j >= 0:
                around, flag = self.add_state_to_around(territoryBoard[x + i][y - j])[:]
                if around != None:
                    around_list.append(around)
                if flag:
                    break
            else:
                break

        for i,j in zip(range(1,self._SIZE),range(1,self._SIZE)):
            if x + i < self._SIZE and y + j < self._SIZE:
                around, flag = self.add_state_to_around(territoryBoard[x + i][y + j])[:]
                if around != None:
                    around_list.append(around)
                if flag:
                    break
            else:
                break

        #print around_list

        if around_list.count(self._BLACK) > around_list.count(self._WHITE):
            targetState = self._BT
        else:
            targetState = self._WT

        territoryBoard[x][y] = targetState
        return territoryBoard

    def add_state_to_around(self, state):
        # 改正した（塚原）
        around = None
        isAdded = False
        if state == self._BLACK:  # 黒の状態をいれる
            around=self._BLACK
            isAdded = True
        elif state == self._WHITE:  # 白の状態をいれる
            around=self._WHITE
            isAdded = True
        return [around, isAdded]

    def calcTerritory(self, territoryBoard):
        # 改正した（塚原）
        ret = None
        black_count = 0
        white_count = 0
        for row in territoryBoard:
            black_count += row.count(self._BT)
            white_count += row.count(self._WT)
        ret = (black_count, white_count)
        return ret
    def make_final_captured_board(self ,go_state_obj):
        #final_board = [[float(0) for x in xrange(self._SIZE)] for _ in xrange(self._SIZE)]
        #final_board = [[go_state_obj._board[y][x] if self.rule.valid_move_public(go_state_obj, self.black_player, (x, y)) else float(0) for x in xrange(self.rule._SIZE)] for y in xrange(self.rule._SIZE)]

        for x in xrange(self._SIZE):
            for y in xrange(self._SIZE):
                if go_state_obj._board[y][x] == self._EMPTY:
                    #4方向みて色が混在しているか
                    bl = False
                    wh = False
                    for [i,j] in self._DIRECTIONS:
                        if x + j < self._SIZE and y + i < self._SIZE and x - j > 0 and y - i > 0:
                            if go_state_obj._board[y+i][x+j] == self._WHITE:
                                wh = True
                            elif go_state_obj._board[y+i][x+j] == self._BLACK:
                                bl = True
                    if wh == True and bl == True:
                        white_can_move = False
                        black_can_move =False
                        if self.valid_move_public(go_state_obj, self.black_player, (x, y)):
                            black_can_move = True
                        if self.valid_move_public(go_state_obj, self.white_player, (x, y)):
                            white_can_move = True
                        if (white_can_move == True and black_can_move==False):
                            go_state_obj=self.move_and_return_state(go_state_obj,self.white_player,(x,y))
                            go_state_obj._board[y][x]=self._EMPTY
                        if (white_can_move == False and black_can_move==True):
                            go_state_obj=self.move_and_return_state(go_state_obj, self.black_player, (x, y))
                            go_state_obj._board[y][x] = self._EMPTY

        #go_state_obj._board = final_board
        print sys.stderr.write(str(self.print_board(go_state_obj)))
        return go_state_obj
    def count_territory(self, go_state_obj):
        # 改正した（塚原）
        go_state_obj=self.make_final_captured_board(go_state_obj)
        print sys.stderr.write(str(self.print_board(go_state_obj)))
        territoryBoard = self.create_territory_map(go_state_obj._board)  # territoryBoardの作成
        # print "territoryBoard" + str(territoryBoard)
        score_tuple = self.calcTerritory(territoryBoard)                # Territoryの計算?
        return score_tuple

    def valid_move(self, player, next_move, montecalro=False):
        '''
        :param player: 現在のプレーヤー
        :param next_move: 次の手の座標(x, y)
        :return: 次の手が打てるかどうかを返す(True or False)
        '''
        (x, y) = next_move
        onboard = self._on_board(x, y)
        #emptyの条件判定はisを使用しないこと
        empty = self._board[y][x] == self._EMPTY
        suicide = self._check_suicide(player, next_move)
        eye = self._check_suicide(player.next_player, next_move)
        ko_place = (x, y) not in self._ko_place
        if not montecalro:
            # sys.stderr.write(str( str(next_move) + " is " + " onboard:" + str(onboard) + " empty:" + str(empty) + " suicide:" + str(suicide) + " ko_place:" + str(ko_place) + " [valid_move]"
            return onboard and empty and suicide and ko_place
        else:
            return onboard and empty and suicide and ko_place and eye

    def valid_move_public(self,go_state_obj, player, next_move, montecalro=False):
        '''
        :param player: 現在のプレーヤー
        :param next_move: 次の手の座標(x, y)
        :return: 次の手が打てるかどうかを返す(True or False)
        '''
        self.copy_board_obj(go_state_obj)
        (x, y) = next_move
        onboard = self._on_board(x, y)
        #emptyの条件判定はisを使用しないこと
        empty = self._board[y][x] == self._EMPTY
        suicide = self._check_suicide(player, next_move)
        eye = self._check_suicide(player.next_player, next_move)
        ko_place = (x, y) not in self._ko_place
        if not montecalro:
            # sys.stderr.write(str( str(next_move) + " is " + " onboard:" + str(onboard) + " empty:" + str(empty) + " suicide:" + str(suicide) + " ko_place:" + str(ko_place) + " [valid_move]"
            return onboard and empty and suicide and ko_place
        else:
            print ko_place
            return onboard and empty and suicide and ko_place and eye

    def next_moves(self, go_state_obj, player, flag=False):
        '''
        :param player: 現在のプレーヤー
        :return: 次に打てる全ての手を返す(リスト型)
        '''
        self.copy_board_obj(go_state_obj)
        result = [(x, y) for x in xrange(self._SIZE) for y in xrange(self._SIZE) if self.valid_move(player, (x, y), flag)] + [self._PASS]
        return result

    def next_moves_flat(self, go_state_obj, player, flag=False):
        self.copy_board_obj(go_state_obj)
        result = [[y * self._SIZE + x] for x in xrange(self._SIZE) for y in xrange(self._SIZE) if self.valid_move(player, (x, y), flag)] + [self._PASS]
        return result

    def copy_board_obj(self, go_state_obj):
        self._board = go_state_obj._board
        self._ko_place = go_state_obj._ko_place
        self._turns_since_board = go_state_obj._turns_since_board
        self.turns_num=go_state_obj.turns_num

    def update_turns_since_board(self, go_state_obj, next_move):
        #sys.stderr.write(str( next_move
        #注意！　c++で実装するときはなぜか値が2回ひかれて-2されてしまう現象を直す！

        if next_move == "pass":
            sys.stderr.write("pass")
            pass
        else:
            (xpos, ypos) = next_move
            go_state_obj._turns_since_board[ypos][xpos] = 8
        #TODO
        #turns_board = [[float(0) if go_state_obj._turns_since_board[y][x] < 1.0 or go_state_obj._board[y][x] == self._EMPTY else float(go_state_obj._turns_since_board[y][x] - 1) for x in xrange(self._SIZE)] for y in xrange(self._SIZE)]
        turns_board = [[float(0) if go_state_obj._turns_since_board[y][x] < 1.0 else float(go_state_obj._turns_since_board[y][x] - 1) for x in xrange(self._SIZE)] for y in xrange(self._SIZE)]
        return turns_board

    def move(self, go_state_obj, player, next_move):
        '''
        :param player: 現在のプレーヤー
        :param next_move: 次の手の座標(x, y)
        :return: 次の状態(盤面、コウの位置)を返す
        passの時は同じ状態を返す
        '''
        self.copy_board_obj(go_state_obj)

        self._turns_since_board = self.update_turns_since_board(go_state_obj, next_move)
        self.turns_num+=1
        flips_list = []
        self._ko_place = []
        if next_move == self._PASS:  # passの時の処理
            return GoStateObject([[self._board[y][x] for x in xrange(self._SIZE)] for y in xrange(self._SIZE)], self._ko_place, self._turns_since_board,self.turns_num)
        else:
            flips_list.extend(self._flips(player, next_move))
            if self._check_ko(player, next_move, len(flips_list)):
                self._ko_place.append(flips_list[0])
            self._board[next_move[1]][next_move[0]] = player.player_id
            return GoStateObject([[self._board[y][x] if (x, y) not in flips_list else self._EMPTY for x in xrange(self._SIZE)] for y in xrange(self._SIZE)], self._ko_place, self._turns_since_board,self.turns_num)

    def move_and_return_state(self, go_state_obj, player, next_move):
        '''
        :param player: 現在のプレーヤー
        :param next_move: 次の手の座標(x, y)
        :return: 次の状態(盤面、コウの位置)を返す(探索用)
        passの時は同じ状態を返す
        '''
        aBoard = go_state_obj._board
        self.copy_board_obj(go_state_obj)

        self._turns_since_board = self.update_turns_since_board(go_state_obj, next_move)
        self.turns_num+=1
        #sys.stderr.write(str(self._turns_since_board))

        flips_list = []
        self._ko_place = []
        if next_move == self._PASS:  # passの時の処理
            return GoStateObject([[aBoard[y][x] for x in xrange(self._SIZE)] for y in xrange(self._SIZE)], self._ko_place, self._turns_since_board,self.turns_num)
        else:
            flips_list.extend(self._flips(player, next_move))
            if self._check_ko(player, next_move, len(flips_list)):
                print "_check_ko True!!!!!!!!!!!!"
                print flips_list
                self._ko_place.append(flips_list[0])
            aBoard[next_move[1]][next_move[0]] = player.player_id
            return GoStateObject([[aBoard[y][x] if (x, y) not in flips_list else self._EMPTY for x in xrange(self._SIZE)] for y in xrange(self._SIZE)], self._ko_place,self._turns_since_board,self.turns_num)

    def serialize(self):
        return tuple(self._board[y][x] for x in xrange(self._SIZE)
                     for y in xrange(self._SIZE))

    def _on_board(self, x, y):
        '''
        :param x: 盤面の水平の座標
        :param y: 盤面の垂直の座標
        :return: ボード上にあるかどうか(True or False)
        '''
        if 0 <= x < self._SIZE and 0 <= y < self._SIZE:
            return True
        else:
            return False

    def _check_suicide(self, player, next_move):
        '''
        :param player: 現在のプレーヤー
        :param next_move: 次の手の座標(x, y)
        :return: 次の手が自殺手かどうかを返す(自殺手ならFalse)
        '''
        #sys.stderr.write(str( "[_check_suicide]:" + str(next_move)
        (x, y) = next_move
        retention = self._board[y][x]
        self._board[y][x] = player.player_id
        if self._check_surrounded(player, next_move):
            if len(self._flips(player, next_move)) > 0:
                self._board[y][x] = retention
                return True
            else:
                self._board[y][x] = retention
                return False
        else:
            self._board[y][x] = retention
            return True

    def _check_surrounded(self, player, next_move):
        '''
        :param player: 囲まれてるか判定する石のプレーヤー
        :param next_move: 次の手の座標(x, y)
        :return: 次の手が囲まれているかどうかを返す(囲まれていたらTrue)
        '''
        searched_list = []
        myself = player.player_id
        # opponent = player.next().player_id
        queue = [(next_move)]
        while len(queue) > 0:
            (x0, y0) = queue.pop(0)
            for (dx, dy) in self._DIRECTIONS:
                (x, y) = (x0 + dx, y0 + dy)
                if self._on_board(x, y) and (x, y) not in searched_list:
                    searched_list.append((x, y))
                    if self._board[y][x] == self._EMPTY:
                        return False
                    elif self._board[y][x] == myself:
                        queue.append((x, y))
        #sys.stderr.write(str( "surrounded!:" + str(next_move)
        return True

    def _check_ko(self, player, stone_pos, flip_num):
        myself = player.player_id
        # opponent = player.next().player_id
        (x0, y0) = stone_pos
        for (dx, dy) in self._DIRECTIONS:
            (x, y) = (x0 + dx, y0 + dy)
            if not self._on_board(x, y):
                continue
            if self._board[y][x] == myself:
                return False
        if flip_num == 1:
            return True
        else:
            return False

    def _flips(self, player, next_move):
        '''
        :param player: 現在のプレーヤー
        :param next_move: 次の手の座標(x, y)
        :return: 次の手を打って取れる全ての石の座標を返す(リスト型)
        実際に石を取る処理は_flip関数に任せてます。_flips関数はその前処理と実際に取れる石リストを返しています。
        '''
        myself = player.player_id
        opponent = player.next_player_func().player_id
        flips = []
        (x0, y0) = next_move
        self._board[y0][x0] = myself
        for (dx, dy) in self._DIRECTIONS:
            (x, y) = (x0 + dx, y0 + dy)
            if not self._on_board(x, y) \
                    or self._board[y][x] == myself \
                    or self._board[y][x] == self._EMPTY \
                    or not self._check_surrounded(player.next_player_func(), (x, y)):
                continue
            elif self._board[y][x] == opponent:
                flips.extend(self._flip(player.next_player_func(), (x, y)))
        self._board[y0][x0] = self._EMPTY
        #sys.stderr.write(str( "flips¥n"
        #sys.stderr.write(str( flips
        return flips

    def _flip(self, player, stone_pos, searched_list=None):
        '''
        :param player: 石を取るプレーヤー
        :param stone_pos: 現在の処理する石の座標
        :param searched_list: 探索した石の座標リスト
        :return: 取れる石の座標を返す(リスト型)
        '''
        myself = player.player_id
        # opponent = player.next().player_id
        result = []
        result.append(stone_pos)
        (x0, y0) = stone_pos
        if searched_list is None:
            searched_list = []
        elif not (x0, y0) in searched_list:
            searched_list.append((x0, y0))
        for (dx, dy) in self._DIRECTIONS:
            (x, y) = (x0 + dx, y0 + dy)
            if not self._on_board(x, y):
                continue
            if self._board[y][x] == myself and (x, y) not in searched_list:
                result.extend(self._flip(player, (x, y), searched_list))
        #sys.stderr.write(str( "result:¥n"
        #sys.stderr.write(str( result

        return result

    def __str__(self):
        # 盤面を表示する
        s = {-1: '.',  # empty
             0: 'x',  # player 0
             1: 'o',  # player 1
             }
        header = '  '
        for i in xrange(self._SIZE):
            header = header + str(i % 10) + ''
        rows = [header]
        for y in xrange(self._SIZE):
            rows.append('  %s %d' % (
                ''.join(s[self._board[y][x]] for x in xrange(self._SIZE)), y))
        return '\n'.join(rows)

    def get_board(self):
        return self._board

    def print_board(self, go_state_obj):
        # 盤面を表示する go_state_objに対応した関数
        s = {-1: '.',  # empty
             0: 'x',  # player 0
             1: 'o',  # player 1
             }
        header = '  '
        for i in xrange(self._SIZE):
            header = header + str(i % 10) + ''
        rows = [header]
        for y in xrange(self._SIZE):
            rows.append('  %s %d' % (
                ''.join(s[go_state_obj._board[y][x]] for x in xrange(self._SIZE)), y))
        return '\n'.join(rows) + '\n\n\n'


def _main():
    # 地を数えるテスト
    from game import Player

    players = [Player(0.0, 'human'), Player(1.0, 'human')]
    players[0].next_player = players[1]
    players[1].next_player = players[0]
    player = players[0]
    rule = Go()
    e = -1.0
    x = 0.0
    o = 1.0
    test1_board = [
        [e, e, e, e, e],
        [e, e, e, e, e],
        [o, o, o, o, o],
        [x, x, x, x, o],
        [e, x, e, x, o]
    ]
    test2_board = [
        [e, e, e, e, e],
        [e, e, e, e, e],
        [o, o, e, e, e],
        [x, x, o, e, e],
        [e, x, o, e, e]
    ]
    test3_board = [
        [e, e, e, e, e],
        [e, e, e, e, e],
        [o, o, e, e, e],
        [x, o, e, e, e],
        [x, e, o, e, e]
    ]
    go_state_obj=GoStateObject()
    go_state_obj._board = test1_board
    # GoStateObject(board=test1_board , ko_place=, turns_since_board=None, turns_num=None)
    print go_state_obj._board
    result = rule.count_territory(go_state_obj)
    print result

    go_state_obj._board =test2_board
    print go_state_obj._board
    result = rule.count_territory(go_state_obj)
    print result

    go_state_obj._board=test3_board
    print go_state_obj._board
    result = rule.count_territory(go_state_obj)
    print result

if __name__ == '__main__':
    _main()
