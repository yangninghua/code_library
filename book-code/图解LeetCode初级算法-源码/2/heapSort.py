#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/2/heapSort.py
#  Project: /Users/king/Python初级算法/code/2
#  Created Date: 2018/10/31
#  Author: hstking hst_king@hotmail.com




from randomList import randomList
import timeit

iList = randomList(20)


def heapSort(iList):
    pass


if __name__ == "__main__":
    print(iList)
    print(heapSortSort(iList))
    print(timeit.timeit("heapSort(iList)", "from __main__ import heapSort,iList", number=100))