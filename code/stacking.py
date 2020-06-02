import pandas as pd
import os
data ={}
root_path = "./submit/submit_stack/"
for i in os.listdir(root_path):
    data[i]  = pd.read_csv(root_path+i,header=None)[1]

data = pd.DataFrame(data)
submit = pd.read_csv(root_path+i,header=None)



submit[1] = data.mode(axis=1)
x = data.apply(lambda x:len(x[x=="刺网"]),axis=1)
submit[1] = data.mode(axis=1)
submit.loc[x>=1,1] ="刺网"
submit.to_csv("./submit/stack_5seed_change.csv",index=None,header=None)