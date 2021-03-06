#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/3/sequentialSearch.py
#  Project: /Users/king/Python初级算法/code/3
#  Created Date: 2018/12/17
#  Author: hstking hst_king@hotmail.com


from randomList import randomList
from quickSort import quickSort
import random

iList = quickSort(randomList(20))

def sequentialSearch(iList, key):
    print("iList = %s" %str(iList))
    print("Find The number : %d" %key)
    iLen = len(iList)
    for i in range(iLen):
        if iList[i] == key:
            return i
    return -1

if __name__ == "__main__":
    keys = [random.choice(iList), random.randrange(min(iList), max(iList))]
    for key in keys:
        res = sequentialSearch(iList, key)
        if res >= 0:
            print("%d is in the list, index is : %d\n" %(key, res))
        else:
            print("%d is not in the list\n" %key)
