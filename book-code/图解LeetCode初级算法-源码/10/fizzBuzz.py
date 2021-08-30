#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/10/fizzBuzz.py
#  Project: /Users/king/Python初级算法/code/10
#  Created Date: 2019/03/11
#  Author: hstking hst_king@hotmail.com




def fizzBuzz(n):
    rList = []
    for i in range(1, n+1):
        if i%3 == 0 and i%5 == 0:
            rList.append("FizzBuzz")
        elif i%3 == 0:
            rList.append("Fizz")
        elif i%5 == 0:
            rList.append("Buzz")
        else:
            rList.append(str(i))
    return rList

if __name__ == "__main__":
    n = 5
    rList = fizzBuzz(n)
    for s in rList:
        print(s)