#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/4/moveZeros_last.py
#  Project: /Users/king/Python初级算法/code/4
#  Created Date: 2019/01/26
#  Author: hstking hst_king@hotmail.com


def moveZeroes(nums):
    for n in nums:
        if n == 0:
            nums.remove(0)
            nums.append(0)
    return


    # left = 0
    # right = 0
    # while right < len(nums) - 1:
    #     if nums[left] == 0:
    #         nums[left:-1] = nums[left + 1:]
    #         nums[-1] = 0
    #         right += 1
    #     else:
    #         left += 1
    #         right += 1
    # return 




if __name__ == "__main__":
    nums = [0, 1, 0, 3, 12]
    print("before nums = %s" %nums)
    moveZeroes(nums)
    print("after nums = %s" %nums)