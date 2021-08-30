#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/8/merge.py
#  Project: /Users/king/Python初级算法/code/8
#  Created Date: 2019/03/08
#  Author: hstking hst_king@hotmail.com



def merge(nums1, m, nums2, n):
    # nums1[m: m+n] = nums2[:n]
    # for right in range(m, m+n):
    #     target = nums1[right]
    #     for left in range(0, right):
    #         if target <= nums1[left]:
    #             nums1[left+1:right+1] = nums1[left:right] #使用Python的切片赋值
    #             nums1[left] = target
    #             break        
    # return

    for i in range(n):
        nums1[m + i] = nums2[i]
    nums1.sort()
    return


if __name__ == "__main__":
    # nums1 = [1, 2, 3, 0, 0, 0]
    # nums2 = [4, 5, 6]
    # m = 3
    # n = 3
    # nums1 = [2, 0]
    # nums2 = [1]
    # m = 1
    # n = 1
    nums1 = [4, 0, 0, 0, 0, 0]
    nums2 = [1, 2, 3, 5, 6]
    m = 1
    n = 5
    print("before nums1 = %s" %(str(nums1)))
    print("before nums2 = %s" %(str(nums2)))
    merge(nums1, m, nums2, n)
    print("after nums1 = %s" %(str(nums1)))