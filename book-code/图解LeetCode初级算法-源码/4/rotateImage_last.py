#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/4/rotateImage_last.py
#  Project: /Users/king/Python初级算法/code/4
#  Created Date: 2019/01/30
#  Author: hstking hst_king@hotmail.com


def rotate(matrix):
    '''二维数组顺时针旋转90°可以等同于将一个二维数组转置（行列互换），然后每行翻转。 '''
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for i in range(len(matrix)):
        matrix[i].reverse()
    return


if __name__ == "__main__":
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    print("beforte matrix:")
    for subList in matrix:
        print(subList)
    rotate(matrix)
    print("after maxtrix:")
    for subList in matrix:
        print(subList)