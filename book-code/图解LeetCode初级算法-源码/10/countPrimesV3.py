#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/10/countPrimesV3.py
#  Project: /Users/king/Python初级算法/code/10
#  Created Date: 2018/10/23
#  Author: hstking hst_king@hotmail.com

import timeit

def countPrimes(n):
    '''埃拉托斯特尼筛法获取质数 '''
    if n<3:
        return 0 # n<3时是没有质数的
    boolList = [True]*n
    boolList[0] = False
    boolList[1] = False
    for i in range(2, int(pow(n, 0.5)) + 1):
        if boolList[i]: #如果boolList[i]的值已经是False了，那说明在之前比较小的数的倍数已经筛选过一次了。可以不用再次筛选了。
            boolList[i::i] = [False]*len(boolList[i::i])
            boolList[i] = True
    # primesList = []
    # i = 0
    # while i < n:
    #     if boolList[i]:
    #         primesList.append(i)
    #     i += 1
    # print(primesList)
    return sum(boolList)

if __name__ == "__main__":
    print(timeit.timeit("countPrimes(10000)", "from __main__ import countPrimes", number=10))
    print(countPrimes(10000))