#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/11/missingNumber.py
#  Project: /Users/king/Python初级算法/code/11
#  Created Date: 2019/03/12
#  Author: hstking hst_king@hotmail.com




def missingNumber(nums):
    try:
        n = (set(range(len(nums) + 1)) - set(nums)).pop()
    except KeyError as e:
        return
    else:
        return n

if __name__ == "__main__":
    nums = [0]
    n = missingNumber(nums)
    print("The missing number is %d" %(n))