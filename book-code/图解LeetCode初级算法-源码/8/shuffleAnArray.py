#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/10/shuffleAnArray.py
#  Project: /Users/king/Python初级算法/code/10
#  Created Date: 2019/03/11
#  Author: hstking hst_king@hotmail.com

import random
import copy

class Solution(object):
    def __init__(self, nums):
        self.nums = nums[:]
        self.numsCopy = nums[:]

    def reset(self):
        self.nums = self.numsCopy
        return self.nums

    def shuffle(self):
        sList = self.nums[:]
        rList = []
        while sList:
            val = random.choice(sList)
            rList.append(val)
            sList.remove(val)
        return rList

        # result = []
        # temp = copy.deepcopy(self.nums) 
        # while len(temp) != 0:
        #     toInsert = temp[random.randint(0,len(temp)-1)]
        #     result.append(toInsert)
        #     temp.remove(toInsert)
        # return result


def showArray(nums):
    print("nums = %s" %(str(nums)))

if __name__ == "__main__":
    nums = [1, 2, 3]
    array = Solution(nums)
    param_1 = array.reset()
    showArray(param_1)
    param_2 = array.shuffle()
    showArray(param_2)