import pandas as pd
import os
root_path = "./submit/result_2020-02-22-11-57-24.csv"
sub = pd.read_csv(root_path,header=None)


sub2 = pd.read_csv("../复现代码/submit/决赛897__result_0.92512_2020-02-21-16-48-55.csv",header=None)

# sub2 = pd.read_csv("../submit_onlin/result_2.6_test放进训练集.csv",header=None)
print('-------------------------------------------------')
print("----------B榜897---------")
print(sub2[1].value_counts(1))
print("----------新的---------")
print(sub[1].value_counts(1))
print('-------------------------------------------------')
print("----------B榜897---------")
print(sub2[sub2[1]!=sub[1]][1].value_counts(1))
print("----------新的---------")
print(sub[sub2[1]!=sub[1]][1].value_counts(1))
print('-------------------------------------------------')
print("----------变化---------")
print(sub[sub2[1]!=sub[1]][1].value_counts(1) -sub2[sub2[1]!=sub[1]][1].value_counts(1))
print('-------------------------------------------------')
print(sub2[sub2[1]!=sub[1]][1].shape[0])