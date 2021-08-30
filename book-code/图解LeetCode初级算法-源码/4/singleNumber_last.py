#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/4/singleNumber_last.py
#  Project: /Users/king/Python初级算法/code/4
#  Created Date: 2019/01/25
#  Author: hstking hst_king@hotmail.com


def singleNumber(nums):
    # nums.sort()
    # length = len(nums)
    # if length < 3:
    #     return nums[0]
    # i = 0
    # while i < length - 2 :
    #     if nums[i] != nums[i+1]:
    #         return nums[i]
    #     else:
    #         i += 2
    # return nums[-1]

    # nums.sort()
    # n = list(set(nums[::2]) - set(nums[1::2]))[0]
    # return n

    n = 0
    for i in nums:
        n ^= i
    return n

if __name__ == "__main__":
    # nums = [4, 1, 2, 1, 2]
    nums = [17,12,5,-6,12,4,17,-5,2,-3,2,4,5,16,-3,-4,15,15,-4,-5,-6]
    sn = singleNumber(nums)
    print("the Single number is : %d" %(sn))