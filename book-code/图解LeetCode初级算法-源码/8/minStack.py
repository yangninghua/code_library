#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/8/minStack.py
#  Project: /Users/king/Python初级算法/code/8
#  Created Date: 2019/03/11
#  Author: hstking hst_king@hotmail.com




class MinStack(object):
    def __init__(self):
        self.stack = []

    def push(self, x):
        self.stack.append(x)

    def pop(self):
        try:
            self.stack.pop():
        except Exception as e:
            pass

    def top(self):
        try:
            x = self.stack[-1]
        except Exception as e:
            return None
        else:
            return x

    def getMin(self):
        try:
            mi = min(self.stack)
        except Exception as e:
            return None
        else:
            return mi


if __name__ == "__main__":
    MS = MinStack()
    MS.push(x)
    MS.pop()
    param_3 = MS.top()
    param_4 = MS.getMin()