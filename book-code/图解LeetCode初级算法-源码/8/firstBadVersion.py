#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/8/firstBadVersion.py
#  Project: /Users/king/Python初级算法/code/8
#  Created Date: 2019/03/08
#  Author: hstking hst_king@hotmail.com


def firstBadVersion(n):
    #折半查找--二分法查找
    left = 1
    right = n
    while left <= n:
        mid = (left + right) //2
        if isBadVersion(mid):
            if isBadVersion(mid - 1):
                right = mid
            else:
                return mid
        else:
            if isBadVersion(mid + 1):
                return mid + 1
            else:
                left = mid

if __name__ == "__main__":
    n = 5
    