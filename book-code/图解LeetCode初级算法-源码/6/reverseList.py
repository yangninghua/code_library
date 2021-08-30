#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/6/reverseList.py
#  Project: /Users/king/Python初级算法/code/6
#  Created Date: 2019/03/03
#  Author: hstking hst_king@hotmail.com


from createNode import *

def reverseList(head: ListNode) -> ListNode:
    # valList = []
    # pointer = head
    # while pointer:
    #     valList.append(pointer.val)
    #     pointer = pointer.next
    # pointer = head
    # while valList:
    #     pointer.val = valList.pop()
    #     pointer = pointer.next
    # return head

    if head == None:
        return None
    left = right = head
    if right.next == None:
        return head
    else:
        right = right.next
        left.next = None
    while right != None:
        head = right
        right = right.next
        head.next = left
        left = head
    return head
    
    


if __name__ == "__main__":
    head = createChain(4)
    showChain(head)
    rHead = reverseList(head)
    showChain(rHead)