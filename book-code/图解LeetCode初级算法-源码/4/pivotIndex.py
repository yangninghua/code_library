#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/4/pivotIndex.py
#  Project: /Users/king/Python初级算法/code/4
#  Created Date: 2019/03/19
#  Author: hstking hst_king@hotmail.com




def pivotIndex(nums):
    # left = 0
    # right = len(nums) - 1
    # sumLeft = 0
    # sumRight = 0
    # while right > left:
    #     if sumLeft == sumRight:
    #         sumLeft += nums[left]
    #         left += 1
    #         sumRight += nums[right]
    #         right -= 1
    #     elif sumLeft > sumRight:
    #         sumRight += nums[right]
    #         right -= 1
    #     else:
    #         sumLeft += nums[left]
    #         left += 1
    # if right == left:
    #     return left
    # else:
    #     return -1
    #这个只能对应正数列表


    if len(nums) < 3:
        return -1
    sum = 0
    for n in nums:
        sum += n
    halfSum = 0
    for i in range(len(nums)):
        if halfSum == (sum - nums[i]) / 2:
            return i
        else:
            halfSum += nums[i]
    return -1


if __name__ == "__main__":
    # nums = [1, 7, 3, 6, 5, 6]
    # nums = [-1, -1, -1, -1, -1, 0]
    # nums = [-1, -1, -1, -1, -1, -1]
    nums = [-1, -1, 0, 1, 1, 0]
    pIndex = pivotIndex(nums)
    print("The pivot index is : %d" %pIndex)