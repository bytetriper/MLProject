from matplotlib import pyplot as plt
# read from loss_lost.pkl 
import pickle
import numpy as np
import os
from math import log2,tan,pi
import scienceplots
plt.style.use('science')
with open('loss_list_ViT.pkl', 'rb') as f:
    loss_list = pickle.load(f)
# f is a list of (tensor,tensor)
def test_fit(x,y):
    return -log2(1-x*0.999)/(y+1e-7)
t=range(len(loss_list))
fig, ax1 = plt.subplots()
data1= [loss[0].item() for loss in loss_list]
data2= [loss[1].item() for loss in loss_list]
data2= [(data2[0]-data)/data2[0] for data in data2]
data1= [(data1[0]-data)/data1[0] for data in data1]
exp_data= [test_fit(data2[i],data1[i]) for i in range(len(data1))]
color = 'tab:blue'
ax1.set_ylabel(r'$\Delta \mathcal{S}$', color=color)
ax1.set_xlabel(r'$\Delta Acc$')
ax1.plot(data2, data1, color=color)
ax1.tick_params(axis='y', labelcolor=color)
plt.savefig('./list.png',bbox_inches='tight')
plt.savefig('./list.svg',bbox_inches='tight')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylabel('SDM', color=color)  # we already handled the x-label with ax1
ax2.set_ybound(0,10)
color=['red' if data2[i]<0.6 else 'g' if data2[i]<0.997 else 'b' for i in range(len(data2)) ]
color='b'
ax2.scatter(data2,exp_data ,color=color)
print(exp_data[-1])

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('./double.pdf',bbox_inches='tight')
plt.savefig('./double.png',bbox_inches='tight')
