#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/11/hammingWeight.py
#  Project: /Users/king/Python初级算法/code/11
#  Created Date: 2019/03/12
#  Author: hstking hst_king@hotmail.com


def hammingWeight(n):
    bList = list(bin(n))
    # sum = 0
    # while bList:
    #     c = bList.pop()
    #     if c == '1':
    #         sum += 1
    # return sum
    return bList.count('1')


if __name__ == "__main__":
    n = 0b00000000000000000000000000001011
    print("The hamming weight is : %d" % (hammingWeight(n)))
