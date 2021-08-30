#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/5/countAndSay.py
#  Project: /Users/king/Python初级算法/code/5
#  Created Date: 2019/02/14
#  Author: hstking hst_king@hotmail.com




def countAndSay(n):
    if n not in range(1, 31):
        return "the number is error, quit ..."
    if n == 1:
        return "1"
    rawStr = '1'
    countList = [] #统计个数
    pointer = 0 #指针
    while n > 1:
        if pointer + 1 < len(rawStr): 
            if rawStr[pointer] == rawStr[pointer + 1]: #当相邻两个数字相同时，指针后移一位
                pointer += 1
            else: #当相邻两数不同时
                countList.append(str(pointer + 1)) #个数,因为下标从0开始，所以需要需要加1
                countList.append(rawStr[pointer]) # 统计rawStr[pointer]这个数字和个数
                rawStr = rawStr[pointer + 1:] #截取后面不同的部分字符串，统计下一个不同的数字个数
                pointer = 0 #指针回零，从头开始统计新的字符串
            continue
        else: #旧的rawStr统计完毕，设置新的rawStr
            countList.append(str(pointer + 1))
            countList.append(rawStr[pointer])
            rawStr = "".join(countList)
            countList = []
            pointer = 0
        n -= 1
    return rawStr

if __name__ == "__main__":
    n = 7
    s = countAndSay(n)
    print("The nums is %d, say %s" %(n, s))