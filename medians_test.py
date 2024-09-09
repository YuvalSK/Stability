# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 17:54:25 2024

@author: User
"""
import pandas as pd
from scipy.stats import median_test

df = pd.read_csv("df_total.csv")
   
x = df.loc[df["condition"]==1, "estimation"]
y = df.loc[df["condition"]==2, "estimation"]

res = median_test(x, y, correction = True, lambda_=1, ties='below')

print(f"Pearson's chi-squared 1 vs. 2 pval: {res.pvalue:.5f}")

x = df.loc[df["condition"]==3, "estimation"]
y = df.loc[df["condition"]==4, "estimation"]

res = median_test(x, y, correction = True, lambda_=1, ties='below')

print(f"Pearson's chi-squared 3 vs. 4 pval: {res.pvalue:.5f}")

x = df.loc[df["condition"]==5, "estimation"]
y = df.loc[df["condition"]==6, "estimation"]

res = median_test(x, y, correction = True, lambda_=1, ties='below')

print(f"Pearson's chi-squared 5 vs. 6 pval: {res.pvalue:.5f}")

x = df.loc[df["condition"]==7, "estimation"]
y = df.loc[df["condition"]==8, "estimation"]

res = median_test(x, y, correction = True, lambda_=1, ties='below')

print(f"Pearson's chi-squared 7 vs. 8 pval: {res.pvalue:.5f}")

x = df.loc[df["condition"]==1, "estimation"]
y = df.loc[df["condition"]==5, "estimation"]

res = median_test(x, y, correction = True, lambda_=1, ties='below')

print(f"Pearson's chi-squared 1 vs. 5 pval: {res.pvalue:.5f}")

x = df.loc[df["condition"]==3, "estimation"]
y = df.loc[df["condition"]==7, "estimation"]

res = median_test(x, y, correction = True, lambda_=1, ties='below')

print(f"Pearson's chi-squared 3 vs. 7 pval: {res.pvalue:.5f}")


