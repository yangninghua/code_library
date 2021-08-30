#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/4/threeSum.py
#  Project: /Users/king/Python初级算法/code/4
#  Created Date: 2019/03/19
#  Author: hstking hst_king@hotmail.com


def threeSum(nums):
    # rList = []
    # for i in range(1, len(nums) - 1):
    #     for j in range(i):
    #         if (nums[i] + nums[j]) * -1 in nums[i + 1:]:
    #             subList = [nums[i], nums[j], (nums[i] + nums[j]) * -1]
    #             subList.sort()
    #             if subList not in rList:
    #                 rList.append(subList)
    # return rList
    # time out

    # if len(nums) < 3 or max(nums) < 0 or min(nums) > 0:
    #     return []
    # rList = []
    # nums.sort()
    # left = 0
    # right = len(nums) - 1
    # if nums.count(0) > 2:
    #     rList.append([0, 0, 0])
    # if nums.count(0) > 0: # 一个正数，一个负数， 一个零的情况
    #     while nums[left] < 0 and nums[right] > 0:
    #         if nums[left] + nums[right] == 0:
    #             rList.append([nums[left], 0, nums[right]])
    #             left += 1
    #             right -= 1
    # nums = [i for i in nums if i != 0] #去掉所有的0

    # left = 0
    # right = len(nums) - 1
    # while nums[left + 1] < 0 and nums[right] > 0:
    #     if nums[left] + nums[left + 1] + nums[right] == 0:
    #         rList.append([nums[left], nums[left + 1], nums[right]])
    #         right += 1
    #         left -= 1
    #     elif nums[left] + nums[left + 1] + nums[right] > 0:
    #         right -= 1
    #     else:
    #         left += 1

    # left = 0
    # right = len(nums) - 1
    # while nums[left] < 0 and nums[right - 1] > 0:
    #     if nums[left] + nums[right] + nums[right - 1] == 0:
    #         rList.append([nums[left], nums[right], nums[right - 1]])
    #         right += 1
    #         left -= 1
    #     elif nums[left] + nums[right] + nums[right - 1] > 0:
    #         right -= 1
    #     else:
    #         left += 1
    # return rList
    # time out

    rList = []
    left = 0
    right = len(nums) - 1
    while nums[left] < 0 and nums[right] > 0:
        if nums[left] + nums[right] == 0 and nums.count(0) > 0: #正负相等，列表有0
            rList.append([nums[left], 0, nums[right]])
        


            

if __name__ == "__main__":
    # nums = [-1, 0, 1, 2, -1, 4]
    nums = [-2, 0, 1, 1, 2]
    # nums = [0, 0, 0]
    # nums = [3, -2, 1, 0]
    rList = threeSum(nums)
    print("return List is : %s" %(str(rList)))