#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/5/isAnagram.py
#  Project: /Users/king/Python初级算法/code/5
#  Created Date: 2019/02/10
#  Author: hstking hst_king@hotmail.com




def isAnagram(s, t):
    # 排序比较
    # if len(s) != len(t):
    #     return False
    # sList = list(s)
    # sList.sort()
    # tList = list(t)
    # tList.sort()
    # for i in range(len(s)):
    #     if sList[i] == tList[i]:
    #         continue
    #     else:
    #         return False
    # return True


    #列表化后删除
    if len(s) != len(t):
        return False
    sList = list(s)
    for word in t:
        try:
            sList.remove(word)
        except ValueError as e:
            return False
    return True




if __name__ == "__main__":
    s = "anagram"
    t = "nagaram"
    print(isAnagram(s, t))