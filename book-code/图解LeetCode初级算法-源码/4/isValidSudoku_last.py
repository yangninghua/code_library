#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/4/isValidSudoku_last.py
#  Project: /Users/king/Python初级算法/code/4
#  Created Date: 2019/01/27
#  Author: hstking hst_king@hotmail.com


def isValidSudoku(board):
    rows = [[] for i in range(9)] #转换成行的列表
    columns = [[] for i in range(9)] #转换成按列的列表
    blocks = [[] for i in range(9)] #转换成按块的列表

    for i in range(9):
        for j in range(9):
            if board[i][j] == '.':
                pass
            else:
                rows[i].append(board[i][j])
                columns[j].append(board[i][j])
                blocks[i // 3 * 3 + j // 3].append(board[i][j])
    for B2L in rows, columns, blocks:
        for subList in B2L:
            if not len(subList) == len(set(subList)):
                return False
    return True

if __name__ == "__main__":
    board = [
["5","3",".",".","7",".",".",".","."],
["6",".",".","1","9","5",".",".","."],
[".","9","8",".",".",".",".","6","."],
["8",".",".",".","6",".",".",".","3"],
["4",".",".","8",".","3",".",".","1"],
["7",".",".",".","2",".",".",".","6"],
[".","6",".",".",".",".","2","8","."],
[".",".",".","4","1","9",".",".","5"],
[".",".",".",".","8",".",".","7","9"]
]
    print(isValidSudoku(board))
