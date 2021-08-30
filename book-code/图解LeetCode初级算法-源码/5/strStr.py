#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/5/strStr.py
#  Project: /Users/king/Python初级算法/code/5
#  Created Date: 2019/02/11
#  Author: hstking hst_king@hotmail.com




def strStr(haystack, needle):
    if len(needle) > len(haystack):
        return -1
    if needle == haystack:
        return 0
    length = len(needle)
    pointer = 0
    lenght = len(needle)
    while pointer <= (len(haystack) - len(needle)):
        if haystack[pointer: pointer + length] == needle: # subStr = hystack[pointer: pointer + length]
            return pointer
        else:
            pointer += 1
    return -1
                
        

if __name__ == "__main__":
    # haystack = "hello"
    # needle = "ll"
    haystack = "mississippi"
    needle = "pi"
    print(strStr(haystack, needle))