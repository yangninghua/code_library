#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/4/plusOne_last.py
#  Project: /Users/king/Python初级算法/code/4
#  Created Date: 2019/01/26
#  Author: hstking hst_king@hotmail.com


def plusOne(digits):
    digits = [str(i) for i in digits]
    n = int("".join(digits))
    n += 1
    rList = [int(i) for i in list(str(n))]
    return rList

    # digits.reverse()
    # flag = True # plus one flag
    # for i in range(len(digits)):
    #     if flag == True:
    #         if digits[i] ==9:
    #             digits[i] = 0
    #         else:
    #             digits[i] += 1
    #             flag = False
    # if digits[-1] == 0:
    #     digits.append(1)
    # digits.reverse()
    # return digits


if __name__ == "__main__":
    digits = [1, 2, 3]
    rList = plusOne(digits)
    print("The List plus one is : %s" %rList)