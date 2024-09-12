#herb频数直方图
import matplotlib.pyplot as plt
import pandas as pd
herb_data = pd.read_csv('./Set2Set/idx_fre_herb.csv')
herb_id=herb_data.values[:,0].astype(int)
herb_fre=herb_data.values[:,2].astype(int)
id_list=list(herb_id)
fre_list=list(herb_fre)


plt.bar(id_list, fre_list)

# 设置标题和坐标轴标签
# plt.title('草药ID与频率分布')
plt.xlabel('herb_ID')
plt.ylabel('Frequency')
# plt.savefig('./herb_frequency.png')

# 显示图表
plt.show()