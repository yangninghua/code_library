#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/9/maxSubArray.py
#  Project: /Users/king/Python初级算法/code/9
#  Created Date: 2019/03/09
#  Author: hstking hst_king@hotmail.com



def maxSubArray(nums):
    subSum = 0
    maxSum = nums[0]
    for i in range(len(nums)):
        subSum += nums[i]
        if subSum > maxSum:
            maxSum = subSum
        if subSum < 0:
            subSum = 0
    return maxSum



if __name__ == "__main__":
    nums = [-2,1,-3,4,-1,2,1,-5,4]
    # nums = [-1, -3 ,-5, -2, -4]
    maxSum = maxSubArray(nums)
    print("The max count is : %d" %maxSum)