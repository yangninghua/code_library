#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/6/createNode.py
#  Project: /Users/king/Python初级算法/code/6
#  Created Date: 2019/02/15
#  Author: hstking hst_king@hotmail.com

import random


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

def createChain(n):
    randomList = random.sample(range(1000), n)
    print("randomList = %s" %(str(randomList)))
    pointer = ListNode(randomList.pop(0))
    head = pointer
    while randomList:
        pointer.next = ListNode(randomList.pop(0))
        pointer = pointer.next
    pointer.next = None
    return head

def createChainFromList(Li)-> bool:
    print("crate the Chain from list %s" %(str(Li)))
    pointer = ListNode(Li.pop(0))
    head = pointer
    while Li:
        pointer.next = ListNode(Li.pop(0))
        pointer = pointer.next
    pointer.next = None
    return head

def showChain(head):
    if head == None:
        print("the chain is None ...")
        return None
    while head.next:
        print("%d -> " %head.val, end='')
        head = head.next
    print("%d" %head.val)
    return

def showChainXth(head, x):
    pointer = head
    length = countChainLength(head)
    if x > length:
        print("index error, the chain length is : %d" %x)
        return
    length = x
    while length > 0:
        head = head.next
        length -= 1
    print("The Chain %dth node value is %d" %(x, head.val))
    return

def countChainLength(head):
    n = 0
    while head.next:
        n += 1
        head = head.next
    return n

def insertNode(head, th, val):
    length = countChainLength(head)
    pointer = head
    if th == 0:
        tempNode = ListNode(val)
        head.val , tempNode.val = tempNode.val, head.val
        tempNode.next = head.next
        head.next = tempNode
        return
    if th < length:
        while th > 1:
            pointer = pointer.next
            th -= 1
        tempNode = ListNode(val)
        tempNode.next = pointer.next
        pointer.next = tempNode
    else:
        while length > 0: 
            pointer = pointer.next
            length -= 1
        tempNode = ListNode(val)
        pointer.next = tempNode
        tempNode.next = None
    return

def deleteNode(head, val):
    pointer = head
    if pointer.val == val: #被删除节点在第一个
        head.val = head.next.val
        head.next = head.next.next
        return

    length = countChainLength(pointer)
    while length > 0:
        if pointer.next.val != val:
            pointer = pointer.next
            length -= 1
            continue
        else: 
            if length == 1: #节点是链表的最后一个
                pointer.next = None
            else:#节点在链表中间
                pointer.next = pointer.next.next
            return

def sortNode(head):
    valList = []
    pointer = head
    while pointer:
        valList.append(pointer.val)
        pointer = pointer.next
    valList.sort()
    pointer = head
    while pointer:
        pointer.val = valList.pop(0)
        pointer = pointer.next
    return 


if __name__ == "__main__":
    # head = createChain(5)
    # showChain(head)
    # showChainXth(head, 3)

    # insertNode(head,999, 1000)
    # showChain(head)

    # deleteNode(head, 1000)
    # showChain(head)

    # sortNode(head)
    # showChain(head)

    Li = [0, 0]
    head = createChainFromList(Li)
    showChain(head)