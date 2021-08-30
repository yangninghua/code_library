#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/2/bubbleSort.py
#  Project: /Users/king/Python初级算法/code/2
#  Created Date: 2018/10/26
#  Author: hstking hst_king@hotmail.com


from randomList import randomList
import timeit

iList = randomList(20)

def bubbleSort(iList):
    '''冒泡排序 '''
    if len(iList) <= 1:
        return iList
    for i in range(1, len(iList)):
        for j in range(0, len(iList)-i): 
            if iList[j] >= iList[j+1]: #比较相邻两数的大小
                iList[j], iList[j+1] = iList[j+1], iList[j] #将大数交换到靠后的位置
        # print("第 %d 轮排序结果:" %i, end="")
        # print(iList)
    return iList

if __name__ == "__main__":
    print(iList)
    print(bubbleSort(iList))
    print(timeit.timeit("bubbleSort(iList)", "from __main__ import bubbleSort,iList", number=100))