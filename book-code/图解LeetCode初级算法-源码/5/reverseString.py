#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/5/reverseString.py
#  Project: /Users/king/Python初级算法/code/5
#  Created Date: 2019/02/10
#  Author: hstking hst_king@hotmail.com



def reverseString(s):
    # s.reverse()
    # return
    #利用Python列表特性 

 

    length = len(s)
    if length < 2:
        return
    for i in range(length//2):
        s[i], s[length - i - 1] = s[length - i - 1] , s[i]
    return


if __name__ == "__main__":
    s = ['h', 'e', 'l', 'l', 'o']
    print("before s = " + str(s))
    reverseString(s)
    print("after s = "  + str(s))