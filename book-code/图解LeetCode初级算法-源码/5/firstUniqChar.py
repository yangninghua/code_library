#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/5/firstUniqChar.py
#  Project: /Users/king/Python初级算法/code/5
#  Created Date: 2019/02/10
#  Author: hstking hst_king@hotmail.com




def firstUniqChar(s):
    if len(s) == 1:
        return 0
    for i in range(len(s)):
        if s[i] not in s[i+1: ] and  s[i] not in s[:i]:
            return i
    return -1

    

    # hash算法,26个字母的字典
    # words = [chr(i) for i in range(97, 123)] #[a - z]
    # values = [0] * 26
    # wordsDic = dict(zip(words, values)) #{'a':0, 'b':0 ...}
    # for word in s:
    #     wordsDic[word] += 1
    # for i in range(len(s)):
    #     if wordsDic[s[i]] == 1:
    #         return i
    # return -1
    

if __name__ == "__main__":
    # s = "leetcode"
    # s = "loveleetcode"
    s = "dddccdbba"
    num = firstUniqChar(s)
    print("the index for first uniq char is : %d" %num)