#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/4/rotate_1.py
#  Project: /Users/king/Python初级算法/code/4
#  Created Date: 2019/01/23
#  Author: hstking hst_king@hotmail.com


def rotate(nums, k):
    # if len(nums) < 2:
    #     return
    # nums.reverse()
    # k = k % len(nums) #避免k大于nums的长度
    # while k > 0:
    #     temp = nums.pop(0)
    #     nums.append(temp)
    #     k -= 1
    # nums.reverse()
    # return 

    # if len(nums) < 2:
    #     return
    # k = k % len(nums)
    # while k > 0:
    #     temp = nums[-1]
    #     nums[1:] = nums[:-1]
    #     nums[0] = temp
    #     k -= 1
    # return

    # if len(nums) < 2:
    #     return 
    # k = k % len(nums)
    # temp = nums[:len(nums) - k] # 1, 2, 3, 4
    # nums[:k] = nums[len(nums) - k:]
    # nums[k:] = temp
    # return

    if len(nums) < 2:
        return 
    k = k % len(nums)
    temp = nums[len(nums) - k:] # 5, 6, 7
    nums[k:] = nums[:len(nums) - k]
    nums[:k] = temp
    return



if __name__ == "__main__":
    nums = [1, 2, 3, 4, 5, 6, 7]
    k = 3
    print("before nums = %s" %nums)
    rotate(nums, k)
    print("after nums = %s" %nums)