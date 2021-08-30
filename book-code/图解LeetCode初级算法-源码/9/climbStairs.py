#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/9/climbStairs.py
#  Project: /Users/king/Python初级算法/code/9
#  Created Date: 2019/03/09
#  Author: hstking hst_king@hotmail.com




def climbStairs(n):
    stepList = [1, 1, 2]
    i = n
    while i > 2:
        stepList.append(stepList[-1] + stepList[-2])
        i -= 1
    return stepList[n]

if __name__ == "__main__":
    n = 5
    print("count need %d step" %climbStairs(n))
    