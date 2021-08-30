#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/5/isPalindrome.py
#  Project: /Users/king/Python初级算法/code/5
#  Created Date: 2019/02/11
#  Author: hstking hst_king@hotmail.com


def isPalindrome(s):
    # 先处理s字符串，将非字母处理掉，大写字母转换成小写字母。
    # if len(s) < 2:
    #     return True
    # sList = []
    # s = s.lower()
    # for word in s:
    #     if word.isalnum():
    #         sList.append(word)
    # n = len(sList) // 2
    # if sList[:n] == sList[:: -1][:n]:
    #     return True
    # else:
    #     return False


    if len(s) < 2:
        return True
    s = s.lower()
    left = 0
    right = len(s) - 1
    while right - left  > 0: 
        if not s[left].isalnum():
            left += 1
            continue
        if not s[right].isalnum():
            right -= 1
            continue
        if s[left] == s[right]:
            left += 1
            right -= 1
        else:
            return False
    return True

if __name__ == "__main__":
    s = "A man, a plan, a canal: Panama"
    # s = "p, h p"
    # s = "0p"
    print(isPalindrome(s))