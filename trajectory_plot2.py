from random import choice,randint
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
#np.set_printoptions(threshold=np.inf)
cmap = mpl.colormaps['Blues']

x_values = np.load('record/xs.npy', allow_pickle=True)#[:-600]
y_values = np.load('record/ys.npy', allow_pickle=True)#[:-600]
all_obs_xvalues = np.load('record/obs_xs.npy', allow_pickle=True)
all_obs_yvalues = np.load('record/obs_ys.npy', allow_pickle=True)

safe_obs_xvalues = np.load('record/safe_obs_xs.npy', allow_pickle=True)
safe_obs_yvalues = np.load('record/safe_obs_ys.npy', allow_pickle=True)
#print(x_values)
for i in range(len(all_obs_xvalues)):
	if (all_obs_xvalues[i] > 1.0):
		all_obs_xvalues[i] = -1 + (all_obs_xvalues[i] - 1)
	elif (all_obs_xvalues[i] < -1.0):
		all_obs_xvalues[i] = 1 + (all_obs_xvalues[i] + 1)


plt.figure(0)
#绘制运动的轨迹图，且颜色由浅入深
point_numbers = np.array(range(len(x_values)))
obs_point_numbers = np.array(range(len(all_obs_xvalues)))
plt.scatter(x_values, y_values, c=point_numbers, cmap=cmap, edgecolors='none', s=15)
plt.scatter(all_obs_xvalues, all_obs_yvalues, c=obs_point_numbers, cmap=plt.cm.Reds, s=40)
plt.scatter(safe_obs_xvalues[-50:], safe_obs_yvalues[-50:], c='red', s=1)
#将起点和终点高亮显示，s=100代表绘制的点的大小
plt.scatter(x_values[0], y_values[0], c='green', s=100)
plt.scatter(x_values[-1], y_values[-1], c='Black', s=100, marker = 's')

plt.axhline(y=1, xmin=-1, xmax=1, color='g',linewidth=4., linestyle='-')
plt.axvline(x=0.995, ymin=-1, ymax=1, color='g',linewidth=4., linestyle='-')
plt.axvline(x=-0.995, ymin=-1, ymax=1, color='g',linewidth=4., linestyle='-')

#plt.fill_between(np.array([-1,1]), np.array([1,1]), np.array([1.05,1.05]))
# 隐藏x、y轴
#plt.axes().get_xaxis().set_visible(True)
#plt.axes().get_yaxis().set_visible(True)
#plt.axes().set_xlim(-1, 1)
#plt.axes().set_ylim(-1.04, 1.04)
#plt.xticks([-1.0, -0.5, 0, 0.5, 1.0])
#plt.yticks([-1.0, -0.5, 0, 0.5, 1.0])
#显示运动轨迹图
#plt.show()
#plt.savefig("/Users/zheng/Desktop/Research/Week1/success.png", dpi=600, format='png')


# plt.figure(1)
# #plot the out, yes, valid from AdamBA
# #out_s = np.load('src/out_s.npy', allow_pickle=True)
# yes_s = np.load('src/yes_s.npy', allow_pickle=True)
# valid_s = np.load('src/valid_s.npy', allow_pickle=True)
# x = np.linspace(-1,1,len(valid_s))

# plt.figure(2)
# #plot the out, yes, valid from AdamBA
# #out_s = np.load('src/out_s.npy', allow_pickle=True)
# phi_ori = np.load('record/phi_ori_col.npy', allow_pickle=True)
# phi_AdamBA = np.load('record/phi_AdamBA_col.npy', allow_pickle=True)
# x1 = np.arange(0,len(phi_AdamBA),1)


# #plt.scatter(x,out_s)
# plt.scatter(x1,phi_AdamBA,marker = '^', label = 'phi_AdamBA')
# plt.scatter(x1,phi_ori,marker = 'o', label = 'phi_ori')
plt.show()