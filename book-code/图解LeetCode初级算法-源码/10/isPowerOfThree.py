#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/10/isPowerOfThree.py
#  Project: /Users/king/Python初级算法/code/10
#  Created Date: 2019/03/11
#  Author: hstking hst_king@hotmail.com




def isPowerOfThree(n):
    power3 = lambda i: pow(3, i)
    i = 0
    while power3(i) == n:
        if power3(i) == n:
            return True
        else:
            i += 1
    return False

if __name__ == "__main__":
    n = 15
    print("n = %d, return %s" %(n, isPowerOfThree(n)))