#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/9/rob.py
#  Project: /Users/king/Python初级算法/code/9
#  Created Date: 2019/03/09
#  Author: hstking hst_king@hotmail.com


def rob(nums):
    if len(nums) == 0:
        return 0
    if len(nums) == 1:
        return nums[0]
    maxCash = [nums[0], max(nums[0], nums[1])]
    for i in range(2, len(nums)):
        maxCash.append(max(nums[i] + maxCash[i - 2], maxCash[i - 1]))
    # print(maxCash)
    return maxCash[-1]


if __name__ == "__main__":
    # nums = [1,2,3,1]
    nums = [2, 7, 9, 3, 1]
    maxCash = rob(nums)
    print("get max cash is %d" %maxCash)