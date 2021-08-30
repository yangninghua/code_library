#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/11/reverseBits.py
#  Project: /Users/king/Python初级算法/code/11
#  Created Date: 2019/03/12
#  Author: hstking hst_king@hotmail.com




def reverseBits(n):
    dec2bin = bin(n)
    newBin = dec2bin[::-1][:-2]
    if len(newBin) < 32:
        newBin += '0'*(32-len(newBin))
    newDec = int(newBin, 2)
    return newDec

if __name__ == "__main__":
    n = 43261596
    print(reverseBits(n))