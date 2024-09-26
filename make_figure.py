import os
import matplotlib.pyplot as plt
import torch

corruptions = os.listdir('corruption_data')
corruptions1 = ['clean test set']+corruptions
results = torch.load('stuff/results.pt')
results = torch.tensor(results).reshape(20, 4).T

labels = [
    (2, 'BatchNorm'),
    (3, 'BatchNorm (reset)'),
    (0, 'Norm-Free'),
    (1, 'Norm-Free (pseudo-reset)'),
]

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.figure(figsize=(6.5, 5.5))
for i, k in labels:
    plt.plot(results[i], label=k,
             marker=['^', 'd', 'o', 'P'][i], markersize=6,
             linestyle=['solid', 'dashed'][int('reset' in k)])
plt.xticks(list(range(len(corruptions1))), corruptions1, rotation='vertical', fontsize=12)
plt.legend(fontsize=11, loc='lower right') # bruh moment
plt.ylabel('Accuracy', fontsize=14)
plt.xlabel('Evaluation distribution', fontsize=14)
plt.title('BatchNorm and Norm-Free networks do not have\nsignificantly different domain adaptation behavior',
          fontsize=13)
plt.tight_layout()
plt.savefig('stuff/figure.png', dpi=200)
plt.show()

