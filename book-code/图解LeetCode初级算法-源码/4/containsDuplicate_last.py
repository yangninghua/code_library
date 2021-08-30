#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/4/containsDuplicate.py
#  Project: /Users/king/Python初级算法/code/4
#  Created Date: 2019/01/24
#  Author: hstking hst_king@hotmail.com


def containsDuplicate(nums):
    # for i in range(len(nums) - 1):
    #     if nums[i] in nums[i+1:]:
    #         return True
    # return False
    # 提交结果：超出时间限制 

    # nums.sort()
    # for i in range(len(nums) - 1):
    #     if nums[i] == nums[i + 1]:
    #         return True
    # return False

    if len(set(nums)) == len(nums):
        return False
    else:
        return True

if __name__ == "__main__":
    nums = [1, 2, 3, 4, 4]
    print(containsDuplicate(nums))