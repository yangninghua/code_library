#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/2/crateRandomList.py
#  Project: /Users/king/Python初级算法/code/2
#  Created Date: 2018/10/26
#  Author: hstking hst_king@hotmail.com

import random

def randomList(n):
    '''返回一个长度为n的整数列表，数据范围[0,1000) '''
    iList = []
    for i in range(n):
        iList.append(random.randrange(1000))
    return iList

if __name__ == "__main__":
    iList = randomList(10)
    print(iList)