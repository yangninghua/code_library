#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/6/hasCycel.py
#  Project: /Users/king/Python初级算法/code/6
#  Created Date: 2019/03/03
#  Author: hstking hst_king@hotmail.com


from createNode import *

def hasCycle(self, head):
    if head == None:
        return False
    fast = head
    slow = head
    while fast:
        try:
            fast = fast.next.next
            slow = slow.next
        except :
            return False
        if fast == None or slow == None:
            return False
        if fast == slow:
            return True

if __name__ == "__main__":
    pass