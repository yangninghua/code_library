#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/2/shellSort.py
#  Project: /Users/king/Python初级算法/code/2
#  Created Date: 2018/10/26
#  Author: hstking hst_king@hotmail.com


from randomList import randomList
import timeit

iList = randomList(8)

def shellSort(iList):
    if len(iList) <= 1:
        return iList
    # if len(iList) <= 20 :
    #     n = 2
    # else:
    #     n = len(iList)//10 #子序列的个数
    gap = len(iList)
    while gap > 1:
        # gap = gap//n
        gap = gap//2
        for i in range(gap, len(iList)):
            for j in range(i%gap, i, gap):
                if iList[i] < iList[j]:
                    iList[i], iList[j] = iList[j], iList[i]
                    print(iList)
    return iList

if __name__ == "__main__":
    print(iList)
    print(shellSort(iList))
    print(timeit.timeit("shellSort(iList)", "from __main__ import shellSort,iList", number=100))