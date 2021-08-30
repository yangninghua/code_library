#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/6/deleteNode.py
#  Project: /Users/king/Python初级算法/code/6
#  Created Date: 2019/02/15
#  Author: hstking hst_king@hotmail.com


from createNode import *


head = createChain(10)

def deleteNode(node):
    val = node.val
    global head
    pointer = head
    while pointer.next.val != val:
        pointer = pointer.next
    pointer.next = pointer.next.next
    return 


if __name__ == "__main__":
    showChain(head)
    insertChain(head, 4, 55)
    showChain(head)
    node = ListNode(55)
    deleteNode(node)
    showChain(head)
