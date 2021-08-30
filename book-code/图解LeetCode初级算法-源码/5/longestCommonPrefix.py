#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/5/longestCommonPrefix.py
#  Project: /Users/king/Python初级算法/code/5
#  Created Date: 2019/02/15
#  Author: hstking hst_king@hotmail.com




def longestCommonPrefix(strList):
    if len(strList) == 0 or "" in strList:
        return ""
    if len(strList) == 1:
        return strList[0]
    publicWordList = []

    minLength = min([len(st) for st in strList]) #最短字符串的长度

    for i in range(minLength):
        for word in strList:
            publicWordList.append(word[:i+1])
        if len(set(publicWordList)) == 1:
            publicWordList = []
        else:
            return strList[0][:i]
    return strList[0][:minLength]




if __name__ == "__main__":
    # strList = ["flower", "flow", "xx"]
    # strList = ["flower", "flow", "flight"]
    strList = ['c', 'c']
    prefixStr = longestCommonPrefix(strList)
    print("the longest common prefix is : %s" %prefixStr)
