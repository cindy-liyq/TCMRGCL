import numpy as np
result_dict1 = {
    'Precision': [0.85, 0.92, 0.78, 0.89],
    'Recall': [0.91, 0.88, 0.94, 0.85],
    'F1-score': [0.88, 0.90, 0.86, 0.87]
}

result_dict2 = {
    'Precision': [1, 2, 3, 4],
    'Recall': [0.1, 0.08, 0.4, 0.5],
    'F1-score': [0.18, 0.0, 0.06, 0.07]
}
mean_values = [0,0]
best=result_dict1
count=0
for (key1, value1),(key2, value2) in zip(result_dict1.items(),result_dict2.items()):
    mean_values[0] = np.mean(value1)
    mean_values[1] = np.mean(value2)
    if mean_values[0]<mean_values[1]:
        count+=1
    else:
        count-=1
if count>0:
    best = result_dict2
print(best)