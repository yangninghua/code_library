#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/7/levelOrder.py
#  Project: /Users/king/Python初级算法/code/7
#  Created Date: 2019/03/16
#  Author: hstking hst_king@hotmail.com


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def levelOrder(root):
    """
    :type root: TreeNode
    :rtype: List[List[int]]
    """
    rList = []
    subList = []
    if not root:
        return rList
    else:
        cNodeList = [root] #current layer node
    nNodeList = [] #next layer node
    while True:
        if cNodeList:
            node = cNodeList.pop(0)
            subList.append(node.val)
            if node.left and node.right:
                nNodeList.append(node.left)
                nNodeList.append(node.right)
            elif node.left:
                nNodeList.append(node.left)
            elif node.right:
                noNdeList.append(node.right)
            else:
                pass
        else:
            rList.append(subList[:])
            subList = []
            if not nNodeList:
                break
            else:
                cNodeList = nNodeList[:]
                nNodeList = []
    return rList


if __name__ == "__main__":
    # root = TreeNode(1)
    # root.left = TreeNode(2)
    # root.right = TreeNode(3)
    print(levelOrder(root))
