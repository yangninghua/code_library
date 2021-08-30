#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#  File: /Users/king/Python初级算法/code/9/maxProfit.py
#  Project: /Users/king/Python初级算法/code/9
#  Created Date: 2019/03/09
#  Author: hstking hst_king@hotmail.com




def maxProfit(prices):
    profit = 0
    for i in range(len(prices) - 1):
        sub = max(prices[i + 1:]) - prices[i]
        profit = max(sub, profit)
    return profit

    # if len(prices) < 2:
    #     return 0
    # maxList = []
    # for j in range(len(prices) - 1):
    #     for i in range(j + 1, len(prices)):
    #         maxList.append(prices[i] - prices[j])
    # profit = max(maxList)
    # if profit < 0:
    #     return 0
    # else:
    #     return profit



    


if __name__ == "__main__":
    prices = [7, 1, 5, 3, 6, 4]
    # prices = [2,1,2,1,0,1,2]
    profit = maxProfit(prices)
    print("The max profit = %d" %profit)