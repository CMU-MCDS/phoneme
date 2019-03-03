import sys
import numpy as np
import matplotlib.pyplot as plt
MT = [0.9853, 0.9912, 0.9966, 0.9968, 0.997, 0.9978, 0.9978, 0.9987, 0.9987, 0.9987]
EL = [0.892, 0.906, 0.906, 0.906, 0.906, 0.906, 0.906, 0.906, 0.906, 0.906]
POS = [0.8243, 0.8587, 0.8932, 0.8998, 0.9006, 0.9084, 0.9189, 0.9213, 0.9216, 0.927]
DEP = [0.9366, 0.9482, 0.9614, 0.9642, 0.965,  0.965,  0.965,  0.965,  0.965,  0.965]
EL_B = [0.822, 0.842, 0.912, 0.967, 0.967, 0.967, 0.967, 0.967, 0.967, 0.967]
MT_B = [0.956, 0.97, 0.975, 0.978, 0.98, 0.98, 0.98, 0.982, 0.982, 0.983]
POS_B = [0.821, 0.879, 0.883, 0.888, 0.897, 0.908, 0.914, 0.917, 0.925, 0.925]
DEP_B = [0.903, 0.926, 0.934, 0.942, 0.943, 0.952, 0.955, 0.96,  0.977, 0.98]
plt.xticks(np.arange(1, 11, step=1))
np.set_printoptions(precision=4)
#plt.plot(t, s,'-s')
t = range(1,11)
line1, = plt.plot(t,MT,'-s',color='b', label='Machine Translation')
line2, = plt.plot(t,EL,'-^', color='r',label='Entity Linking')
line3, = plt.plot(t,POS,'-o', color='g',label='POS Tagging')
line4, = plt.plot(t,DEP,'-x', color='k',label='Dependency Parsing')

line5, = plt.plot(t,MT_B,'--s',color='b', label='Machine Translation Baseline')
line6, = plt.plot(t,EL_B,'--^',color='r', label='Entity Linking Baseline')
line7, = plt.plot(t,POS_B,'--o',color='g', label='POS Tagging Baseline')
line8, = plt.plot(t,DEP_B,'--x',color='k', label='Dependency Parsing Baseline')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.legend((line1, line2, line3,line4,line5,line6,line7, line8), ('MT LangRank', 'EL LangRank', 'POS LangRank', 'DEP LangRank', 'MT Subword Overlap','EL Genetic','POS Geographic', 'DEP Word Overlap'),loc='lower right')
plt.xlabel('K, number of recommended transfer languages',fontsize=12)
plt.ylabel('Max evaluation score',fontsize=12)
#plt.rcParams.update({'font.size': 30})


#plt.title('Max evaluation score ratio vs. Rank of selected systems')
plt.ylim(0.7, 1) 
plt.savefig("./figMaxEval_20190302_1616.pdf")
plt.show()
