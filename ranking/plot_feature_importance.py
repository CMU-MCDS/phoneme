import matplotlib.pyplot as plt
import numpy as np

total_feature = ['o_w','o_sw','s_tf','s_tk','s_tf/s_tk','ttr_tf','ttr_tk','d_ttr','d_gen','d_syn','d_fea','d_pho','d_inv','d_geo']
#MT_feature = ['o_w','o_sw','s_tf','s_tg','s_tf/s_tg','ttr_tf','ttr_tg','d_ttr','d_gen','d_syn','d_fea','d_pho','d_inv','d_geo']
MT = [168, 140, 117,  82, 182,  99,  59, 129,  54, 127,  89, 32,  77, 130]
MT_Sum = sum(MT)
# EL_feature =  ['o_w','s_tf','s_tg','s_tf/s_tg','d_gen','d_syn','d_fea','d_pho','d_inv','d_geo']
EL = [  4, 0, 74,  36,  31,0,0,0,  17, 105,  27,  46,  31, 117]
EL_Sum = sum(EL)
# POS_feature = ['o_w','s_tf','s_tg','s_tf/s_tg','ttr_tf','ttr_tg','d_ttr','d_gen','d_syn','d_fea','d_pho','d_inv','d_geo']
POS = [109, 0, 96,  40, 193,  93,  64, 129,  30, 105,  74,  64,  92,  96]
POS_Sum = sum(POS)
# Parse_feature = ['o_w','s_tf','s_tk','s_tf/s_tk','ttr_tf','ttr_tk','d_ttr','d_geo','d_gen','d_syn','d_fea','d_inv','d_pho']
Parse = [22.5, 0, 14.433, 8.067, 14.867, 9.667, 8.7, 16.867, 19.867, 11.567, 6.9, 6.867, 8.633, 23.067]
Parse_Sum = sum(Parse)
MT = [ MT[i]/MT_Sum for i in range(len(total_feature))]
EL = [ EL[i]/EL_Sum for i in range(len(total_feature))]
POS = [ POS[i]/POS_Sum for i in range(len(total_feature))]
Parse = [Parse[i]/Parse_Sum for i in range(len(total_feature))]

sum_all = []
for i in range(len(MT)):
    sum_all.append(MT[i]+EL[i]+POS[i]+Parse[i])
print (sum_all)
index_sorted = sorted(range(len(sum_all)), key=lambda k: -sum_all[k])
    
new_MT = []
new_EL = []
new_POS = []
new_Parse = []
new_total_feature = []
for i in range(len(sum_all)):
    new_total_feature.append(total_feature[index_sorted[i]])
    new_MT.append(MT[index_sorted[i]])
    new_EL.append(EL[index_sorted[i]])
    new_POS.append(POS[index_sorted[i]])
    new_Parse.append(Parse[index_sorted[i]])

fig, ax = plt.subplots()
width =0.2
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
y_pos = np.arange(0,len(new_total_feature))
ax.barh(y_pos, new_MT, width,color='#253494', label='MT')
ax.set_yticklabels(new_total_feature)
#ax.barh(y_pos, EL,  color='blue', label='EL')
ax.set_yticks(y_pos)
#ax.set(yticks=y_pos, yticklabels=df.graph, ylim=[2*width - 1, len(df)])


ax.barh(y_pos+width, new_EL, width, color='#2c7fb8', label='EL')
ax.barh(y_pos+2*width, new_POS, width, color='#7fcdbb', label='POS')
ax.barh(y_pos+3*width, new_Parse, width, color='#d8b365', label='DEP')
ax.set(yticks=y_pos+ width, yticklabels=new_total_feature, ylim=[4*width-1, len(total_feature)])





#for i, v in enumerate(MT):
#    ax.text(v + 3, i + .25, str(v), color='red', fontweight='bold')
ax.legend()
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Normalized Importance',fontsize=12)
#ax.set_title('Normalized Importance')
plt.savefig('feature_importance_all_new.pdf')
plt.show()
