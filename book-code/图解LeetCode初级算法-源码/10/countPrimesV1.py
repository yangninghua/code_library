#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/10/countPrimesV1.py
#  Project: /Users/king/Python初级算法/code/10
#  Created Date: 2018/10/18
#  Author: hstking hst_king@hotmail.com

import timeit

def countPrimes(n):
    '''按照质数的定义，用常规的方法来取质数 '''
    primesList = []
    for i in range(2, n+1):
        flag = True
        for divNum in range(2, i): # 从2到i-1，一个一个的除，如果有余数为0的状况，可以确定不是质数，退出循环
            if i % divNum == 0:
                flag = False
                break
        if flag:
            primesList.append(i)
    # print(primesList)
    return len(primesList)

if __name__ == "__main__":
    print(timeit.timeit("countPrimes(10000)", "from __main__ import countPrimes", number=10))
    print(countPrimes(10000))