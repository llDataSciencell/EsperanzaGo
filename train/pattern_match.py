#coding:utf-8

#from go.py import go

'''
board    :[[]]
w_pattern:[[[]]]
b_pattern:[[[]]]
player   :0 or 1 (White or Black)
'''
def pattern_match(board,w_pattern,b_pattern,player):

    BOARD_SIZE = 19
    PATTERN_NUM = 1
    BLOCK_SIZE = 3

    if(player == 0):
        pattern = w_pattern
    else:
        pattern = b_pattern

    match_point = []
    pattern_match_flag = 0

    for i in range(PATTERN_NUM):
        expand_pattern = get_expand_pattern(pattern[i])

        for j in range(BOARD_SIZE-BLOCK_SIZE):
            for k in range(BOARD_SIZE-BLOCK_SIZE):
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
def get_expand_pattern(p):
        expand_pattern = []
        expand_pattern.append(p)
        expand_pattern.append(clockwise(expand_pattern[0]))
        expand_pattern.append(clockwise(expand_pattern[1]))
        expand_pattern.append(clockwise(expand_pattern[2]))
        expand_pattern.append(reverse(expand_pattern[3]))
        expand_pattern.append(clockwise(expand_pattern[4]))
        expand_pattern.append(clockwise(expand_pattern[5]))
        expand_pattern.append(clockwise(expand_pattern[6]))

        return expand_pattern


def clockwise(p):
    q = [[p[2][0],p[1][0],p[0][0]],
         [p[2][1],p[1][1],p[0][1]],
         [p[2][2],p[1][2],p[0][2]] ]

    return q


def reverse(p):
    q = [[p[0][2],p[0][1],p[0][0]],
         [p[1][2],p[1][1],p[1][0]],
         [p[2][2],p[2][1],p[2][0]] ]

    return q


### test function ###
def check_board(b):
    for i in b:
        print(i)


def check_expand_pattern(ep):
    count = 1
    for i in ep:
        print(count)
        for j in i:
            print(j)
        count += 1


if __name__ == "__main__":
    ### test case ###
    O=float(0)  #White
    X=float(1)  #Black
    E=float(-1) #Empty

    board = [
      [E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E],
      [E,E,E,E,O,E,X,E,E,E,E,E,E,E,E,E,E,E,E],
      [E,E,E,E,E,X,E,E,E,E,E,E,E,E,E,E,E,E,E],
      [E,E,E,E,E,O,E,E,E,E,E,E,E,E,E,E,E,E,E],
      [E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E],
      [E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E],
      [E,E,E,X,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E],
      [E,E,E,E,X,O,E,E,E,E,E,E,E,E,E,E,E,E,E],
      [E,E,E,O,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E],
      [E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E],
      [E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E],
      [E,E,E,E,O,O,E,E,E,E,E,E,E,E,E,E,E,E,E],
      [E,E,E,X,O,X,E,E,E,E,E,E,E,E,E,E,E,E,E],
      [E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E],
      [E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E],
      [E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E],
      [E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E],
      [E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E],
      [E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E] ]

    w_pattern = [[[X, E, E],
                  [E, X, O],
                  [O, E, E] ]]

    b_pattern = [[[E, O, O],
                  [X, O, X],
                  [E, E, E] ]]

    player = 1 # White:0, Black:1

    pattern_match(board,w_pattern,b_pattern,player)
