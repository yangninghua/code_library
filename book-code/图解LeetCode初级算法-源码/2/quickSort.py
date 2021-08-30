#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/2/quickSort.py
#  Project: /Users/king/Python初级算法/code/2
#  Created Date: 2018/10/26
#  Author: hstking hst_king@hotmail.com


from randomList import randomList
import timeit

iList = randomList(20)

# def quickSort(iList):
#     if len(iList) <= 1:
#         return iList
#     left = []
#     right = []
#     for i in iList[1:]:
#         if i <= iList[0]:
#             left.append(i)
#         else:
#             right.append(i)
#     return quickSort(left) + [iList[0]] + quickSort(right)

def quickSort(iList):
    if len(iList) <= 1:
        return iList
    left = []
    right = []
    for i in iList[1:]:
        if i <= iList[0]:
            left.append(i)
        else:
            right.append(i)
    return quickSort(left) + [iList[0]] + quickSort(right)

if __name__ == "__main__":
    print(iList)
    print(quickSort(iList))
    print(timeit.timeit("quickSort(iList)", "from __main__ import quickSort,iList", number=100))