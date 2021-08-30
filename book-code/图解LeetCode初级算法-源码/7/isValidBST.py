#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/7/isValidBST.py
#  Project: /Users/king/Python初级算法/code/7
#  Created Date: 2019/03/07
#  Author: hstking hst_king@hotmail.com


from createTreeNode import *

def isValidBST(root):



if __name__ == "__main__":
    root = TreeNode(10)
    root.left = TreeNode(5)
    root.right = TreeNode(15)
    root.right.left = TreeNode(6)
    root.right.right = TreeNode(20)
    print(isValidBST(root))