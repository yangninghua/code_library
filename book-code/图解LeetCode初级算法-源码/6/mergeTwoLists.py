#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/6/mergeTwoLists.py
#  Project: /Users/king/Python初级算法/code/6
#  Created Date: 2019/03/03
#  Author: hstking hst_king@hotmail.com



from createNode import *

def mergeTwoLists(l1, l2):
    if l1 == None and l2 == None: #两条列表都是空列表
        return None
    fakeNode = ListNode(0)
    head = fakeNode
    while l1 != None or l2 != None: #俩调链表遍历完毕
        if l1 == None: #l1遍历完毕
            head.next = l2
            l2 = l2.next
            head = head.next
        elif l2 == None: #l2遍历完毕
            head.next = l1
            l1 = l1.next
            head = head.next
        else:
            if l1.val <= l2.val: #比较l1.val 和 l2.val
                head.next = l1  
                l1 = l1.next
                head = head.next
            else:
                head.next = l2
                l2 = l2.next
                head = head.next
    return fakeNode.next

if __name__ == "__main__":
    l1 = createChain(3)
    sortNode(l1)
    showChain(l1)
    l2 = createChain(3)
    sortNode(l2)
    showChain(l2)
    head = mergeTwoLists(l1, l2)
    showChain(head)
                