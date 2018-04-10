import numpy as np
conv_filters = [[7.7],[6,6],[5,5],[4,4],[3,3],[2,2]]
lr= 0.001
for i,_ in enumerate(conv_filters):
    lr *= 0.98
    print(float(i))