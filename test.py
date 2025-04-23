from scipy.stats import norm,binom
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
file_path=r'C:\Users\etram\Downloads\mid_results.csv'
df=pd.read_csv(file_path)
df.columns=['ID','mark/20']
df=df.set_index('ID')
# del df['mark/25']
std_value=df['mark/20'].std().round(2)
median=df['mark/20'].median().round(2)
full_mark=df[df['mark/20']==20]
print(f'standard Deviation: {std_value}')
print(f'the mean {df["mark/20"].mean().round(2)}')
print(f'the median {median}')

counts,bins,pathc=plt.hist(df,bins=8,edgecolor='black')
for i, count in enumerate(counts):
    x = (bins[i] + bins[i + 1]) / 2 
    y = count   
    plt.text(x, y, str(int(count)), ha='center', va='bottom', fontsize=10)
plt.title('hist of marks')
plt.xlabel('degree')
plt.ylabel('number of student')
plt.show()
df.sample(2,replace=True)