#coding:utf-8
from go import GoStateObject
class Pattern(object):
    def __init__(self):
        #TODO: xはfor文の時などに変数として使わないこと
        #次黒番で統一
        rule = GoStateObject()
        _ = rule._EMPTY
        x = rule._BLACK
        o = rule._WHITE
        B = float(2)  # state is outside of board
        A = float(777)  # any kind of state is ok

        self.pattern=[]
        self.edge_pattern=[]
        self.pattern.append([[[A, o, A],
                              [x, _, x],
                              [A, o, A]],
                              100])
        self.pattern.append([[[o, o, _],
                              [x, _, _],
                              [_, _, A]],
                              100])
        self.pattern.append([[[x, o, _],
                              [o, _, o],
                              [_, x, _]],
                              120])


        self.pattern.append([[[x, o, A],
                              [_, _, x],
                              [A, _, A]],
                              100])

        self.pattern.append([[[x, o, o],
                              [A, _, _],
                              [A, _, x]],
                              50])

        self.pattern.append([[[x, _, o],
                              [A, _, A],
                              [o, x, A]],
                              100])

        self.pattern.append([[[x, _, A],
                              [A, _, A],
                              [A, _, x]],
                              80])

        self.pattern.append([[[A, x, A],
                              [_, _, o],
                              [A, x, A]],
                              70])
        self.pattern.append([[[A, x, A],
                              [o, _, o],
                              [A, x, A]],
                              100])

        self.pattern.append([[[o, _, A],
                              [x, _, x],
                              [A, _, o]],
                              100])

        self.pattern.append([[[o, x, o],
                              [x, _, x],
                              [A, o, A]],
                              130])

        self.pattern.append([[[o, x, _],
                              [x, _, o],
                              [_, A, _]],
                              150])

        self.edge_pattern.append([[[B, _, A],
                                  [B, _, o],
                                  [B, _, x]],
                                  50])


        self.edge_pattern.append([[[B, _, A],
                                  [B, _, o],
                                  [B, x, x]],
                                  20])

