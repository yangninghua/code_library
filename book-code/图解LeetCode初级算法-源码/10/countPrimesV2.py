#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/10/countPrimesV2.py
#  Project: /Users/king/Python初级算法/code/10
#  Created Date: 2018/10/18
#  Author: hstking hst_king@hotmail.com

import timeit

def countPrimes(n):
    '''倍数筛选法获取质数 '''
    if n <= 2:
        return 0
    elif n == 3:
        return 1
    elif n == 4:
        return 2
    primesList = list(range(2, n))
    multList = []
    for i in range(0, n//2 + 1):
        multList += list(range(primesList[i], n+1))[::primesList[i]][1::]
    primesList = list(set(primesList) - set(multList)) # 利用集合过滤掉倍数
    primesList.sort()
    # print(primesList)
    return len(primesList)

if __name__ == "__main__":
    # print(timeit.timeit("countPrimes(10000)", "from __main__ import countPrimes", number=10))
    # print(countPrimes(10000))
    print(countPrimes(5))