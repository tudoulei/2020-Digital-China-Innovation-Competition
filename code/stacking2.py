import pandas as pd
import numpy as np
type_map_rev = {0: '拖网', 1: '围网', 2: '刺网'}

data1 = pd.read_csv("./submit/PROB_softmax_result_0.92512_2020-02-21-16-48-55.csv",header=None).values
data2 = pd.read_csv("./submit/result_93232B_1405_Pro.csv").values[:,2:]

sub2 = pd.read_csv("./submit/result_93232B_1405(1).csv",header=None)
sub2[1] = np.argmax(data1*0.7+data2*0.3,axis=1)
sub2[1] = sub2[1].map(type_map_rev)
sub2.to_csv('submit/stack_B.csv', index=None, header=None)
    