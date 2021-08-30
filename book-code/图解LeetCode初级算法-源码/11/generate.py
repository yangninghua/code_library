#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/11/generate.py
#  Project: /Users/king/Python初级算法/code/11
#  Created Date: 2019/03/12
#  Author: hstking hst_king@hotmail.com



def generate(numRows):
    rList = []
    for i in range(1, numRows + 1):
        tempList = i * [None]
        tempList[0] = 1
        tempList[-1] = 1
        # if i > 2:
        if None in tempList:
            for n in range(1, i - 1):
                tempList[n] = rList[i - 2][n - 1] + rList[i - 2][n]
        rList.append(tempList)
    return rList


if __name__ == "__main__":
    numRows = 5
    print(generate(numRows))