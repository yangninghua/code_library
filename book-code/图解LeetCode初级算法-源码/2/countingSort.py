#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/2/countingSort.py
#  Project: /Users/king/Python初级算法/code/2
#  Created Date: 2018/10/26
#  Author: hstking hst_king@hotmail.com


from randomList import randomList
import timeit

iList = randomList(20)

def countingSort(iList):
    if len(iList) <= 1:
        return iList
    iLen = len(iList)
    rList = [None]*iLen
    for i in range(iLen):
        small = 0 #比基数小的
        same = 0 #与基数相等的
        for j in range(iLen):
            if iList[j] < iList[i]:
                small += 1
            elif iList[j] == iList[i]: #相同的数
                same += 1
        for k in range(small, small+same):
            rList[k] = iList[i]
    return rList

if __name__ == "__main__":
    print(iList)
    print(countingSort(iList))
    print(timeit.timeit("countingSort(iList)", "from __main__ import countingSort,iList", number=100))