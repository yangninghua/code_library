#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/4/maxProfit2.py
#  Project: /Users/king/Python初级算法/code/4
#  Created Date: 2019/01/21
#  Author: hstking hst_king@hotmail.com

def maxProfit(prices):
    maxPro = 0
    i = 1
    while i < len(prices):
        profit = prices[i] - prices[i-1]
        if profit > 0:
            maxPro += profit
        i += 1
    return maxPro

if __name__ == "__main__":
    prices = [7, 1, 5, 3, 6, 4]
    print(maxProfit(prices))