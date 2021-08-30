#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/10/romanToInt.py
#  Project: /Users/king/Python初级算法/code/10
#  Created Date: 2019/03/12
#  Author: hstking hst_king@hotmail.com


def romanToInt(s):
    romanDic = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000
    }
    doubleDic = {
        "IV": 4,
        "IX": 9,
        "XL": 40,
        "XC": 90,
        "CD": 400,
        "CM": 900
    }
    sum = 0 #count number
    i = 0 #pointer
    while i < len(s):
        try:
            s[i:i+2]
        except IndexError as e:
            sum += romanDic[s[i]]
            i += 1
        else:
            if s[i:i+2] in doubleDic.keys():
                sum += doubleDic[s[i:i+2]]
                i += 2
            else:
                sum += romanDic[s[i]]
                i += 1
    return sum

if __name__ == "__main__":
    # s = "III"
    s = "MCMXCIV"
    # s = "IV"
    print(romanToInt(s))