#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/4/removeDuplicates_1.py
#  Project: /Users/king/Python初级算法/code/4
#  Created Date: 2019/01/15
#  Author: hstking hst_king@hotmail.com


def removeDuplicates(nums):
    n = len(set(nums))
    i = 0
    while i < n:
        if nums[i] == nums[i+1]:
            temp = nums[i+1]
            nums[i+1: len(nums) -1] = nums[i+2:]
            nums[-1] = temp
            continue
        else:
            i += 1
    return n

if __name__ == "__main__":
    nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
    print("before nums = %s" %nums)
    print(removeDuplicates(nums))
    print("after nums = %s" %nums)