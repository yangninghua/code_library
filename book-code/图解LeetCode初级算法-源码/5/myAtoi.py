#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/5/myAtoi.py
#  Project: /Users/king/Python初级算法/code/5
#  Created Date: 2019/02/11
#  Author: hstking hst_king@hotmail.com




def myAtoi(s):
    s = s.lstrip() #去开头空格
    if len(s) < 1:
        return 0
    minusFlag = False #假设最终数字非负
    if s[0] in ['+', '-']:
        if s[0] == '+':
            pass
        else:
            minusFlag = True
        s = s[1:] #确定负号后，去除掉负号标记

    if len(s) < 1: #去除正负号后字符串长度
        return 0
    if not s[0].isdigit(): #符号后不是数字
        return 0

    iList = []
    for i in range(len(s)):
        if s[i].isdigit():
            iList.append(s[i])
        else:
            break

    INT_MAX = pow(2, 31) - 1
    INT_MIN = pow(2, 31) * (-1)
    if minusFlag: #测试整数区间
        num = int("".join(iList)) * (-1)
        if num < INT_MIN: 
            num = pow(2, 31) * (-1)
    else:
        num = int("".join(iList))
        if num > INT_MAX:
            num = pow(2, 31) - 1
    return num






if __name__ == "__main__":
    # s = "-4197 with words"
    s = "+-2"
    # s = "words and 987"
    # s = "42"
    num = myAtoi(s)
    print("the string to inteter is : %d" %num)
