#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/7/createTreeNode.py
#  Project: /Users/king/Python初级算法/code/7
#  Created Date: 2019/03/07
#  Author: hstking hst_king@hotmail.com


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def createTreeFromList(tList):
    val = tList.pop(0)
    if val:
        return
    else:
        root.val = val
    val = tList.pop(0)
    if val:
        return
    else:
        root.left.val = val
    val = tList.pop(0)
    if val:
        return
    else:
        root.right.val = val
    # root = TreeNode(tList.pop(0))
    # root.left = createTreeFromList(tList)
    # root.right = createTreeFromList(tList)
    return root

def showMidTree(root):
    if root.val == None:
        print("None -> " , end="")
        return
    else:
        print("%d -> " %root.val, end="")

    showMidTree(root.left)
    showMidTree(root.right)
    return

if __name__ == "__main__":
    # tList = [3, 9, 20, None, None, 15, 7]
    tList = [3, 9, None, None, 20, 15, 7]
    root = createTreeFromList(tList)
    showMidTree(root)