#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/6/isPalindrome.py
#  Project: /Users/king/Python初级算法/code/6
#  Created Date: 2019/03/03
#  Author: hstking hst_king@hotmail.com

from createNode import *


def isPalindrome(head):
    if head == None:
        return True
    valList = []
    while head:
        valList.append(head.val)
        head = head.next
    rList = valList[::-1]
    if rList == valList:
        return True
    else:
        return False
    

if __name__ == '__main__':
    head = createChainFromList([0, 0])
    showChain(head)
    print(isPalindrome(head))
