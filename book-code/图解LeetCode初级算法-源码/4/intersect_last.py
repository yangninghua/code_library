#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/4/intersect_last.py
#  Project: /Users/king/Python初级算法/code/4
#  Created Date: 2019/01/26
#  Author: hstking hst_king@hotmail.com


def intersect(nums1, nums2):
    # rList = []
    # if len(nums1) > len(nums2):
    #     nums1, nums2 = nums2, nums1
    # for n in nums1:
    #     if n in nums2:
    #         rList.append(n)
    #         nums2.remove(n)
    # return rList
    # nums1 比 nums2小很多



    # rList = []
    # nums1.sort()
    # nums2.sort()
    # p1 = 0 #point for nums1
    # p2 = 0 #point for nums2
    # while p1 < len(nums1) and p2 < len(nums2):
    #     if nums1[p1] < nums2[p2]:
    #         p1 += 1
    #     elif nums1[p1] == nums2[p2]:
    #         rList.append(nums1[p1])
    #         p1 += 1
    #         p2 += 1
    #     else:
    #         p2 += 1
    # return rList
    #排序后算法




    # 内存不够
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    rList = []
    left = 0
    right = len(nums1) - 1
    while right < len(nums2):
        temp = nums2[left: right]
        for n in nums1:
            if n in temp:
                rList.append(n)
                temp.remove(n)
        left = right
        if right + len(nums1) >= len(nums2):
            right = len(nums2)
        else:
            right += len(nums1)
    return rLis
    # 分段读取nums2

        




if __name__ == "__main__":
    # nums1 = [4, 9, 5]
    # nums2 = [9, 4, 9, 8, 4]
    nums1 = [1, 2, 2, 1]
    nums2 = [2, 2]
    rList = intersect(nums1, nums2)
    print("The intersection is : %s"  %rList)