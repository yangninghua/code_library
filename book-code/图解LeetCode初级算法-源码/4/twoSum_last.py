#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/4/twoSum_last.py
#  Project: /Users/king/Python初级算法/code/4
#  Created Date: 2019/01/26
#  Author: hstking hst_king@hotmail.com


def twoSum(nums, target):
    # for i in range(len(nums)):
    #     if target - nums[i] in nums:
    #         j = nums.index(target - nums[i])
    #         if i == j:
    #             pass
    #         else:
    #             return [i, j]
    # 利用字典可能会比较快一点

    # for i in range(len(nums) - 1):
    #     for j in range(i+1, len(nums)):
    #         if nums[i] + nums[j] == target:
    #             return [i, j]

    for i in range(len(nums) - 1):
        if target - nums[i] in nums[i+1:]:
            j = nums[i+1:].index(target - nums[i])
            return [i, i + j + 1]


if __name__ == "__main__":
    nums = [2, 7, 11, 15]
    target = 9
    # nums = [3, 3]
    # target = 6
    rList = twoSum(nums, target)
    print("rList = %s" % rList)
