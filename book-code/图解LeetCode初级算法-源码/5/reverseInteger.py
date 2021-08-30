#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/5/reverse.py
#  Project: /Users/king/Python初级算法/code/5
#  Created Date: 2019/02/10
#  Author: hstking hst_king@hotmail.com



def reverse(x):
    # tList = list(str(x))
    # if tList[0] == '-':
    #     rNum = int(''.join(tList[1:][::-1])) * (-1) #列表反转
    # else:
    #     rNum = int(''.join(tList[::-1])) #列表反转
    #
    # if rNum in range(pow(2, 31)* (-1), pow(2, 31) -1):
    #     return rNum
    # else:
    #     return 0

    rList = []
    minus = False #负数标识，为False时表明是正数，True时为负数
    if x < 0:
        minus = True
        x = x * (-1) #将输入的数变成正数

    while x // 10 != 0: #将参数x的绝对值倒序放入列表
        rList.append(str(x % 10))
        x = x // 10
    rList.append(x)

    length = len(rList)
    rNum = 0
    for i in range(length): #列表还原成数字
        rNum += int(rList[i]) * pow(10, length - i - 1)

    if minus: #还原参数的符号
        rNum = rNum * (-1)

    if rNum in range(pow(2, 31)* (-1), pow(2, 31) -1): #判断值区间
        return rNum
    else:
        return 0



if __name__ == "__main__":
    x = 120
    print("numbers = %d" %x)
    rNum = reverse(x)
    print("reverse number = %d" %rNum)