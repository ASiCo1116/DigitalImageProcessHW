import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from IPython import embed

imgs = []
strings = ['', '', '', '']
sets = []
titles = []
for img in glob('./*.64'):
    titles.append(img)
    x = pd.read_csv(img)
    imgs.append(x.values.flatten())

for i, img in enumerate(imgs):
    for s in img:
        strings[i] += s
    sets.append(set(strings[i]))

for i, set_ in enumerate(sets):
    dic = {}

    for s in set_:
        if not s == '\x1a':
            dic[f'{str(s)}'] = strings[i].count(str(s))

    value = list(dic.values())
    key = list(dic.keys())

    sort = sorted(dic.items(), key = lambda x: x[0])

    plt.bar(np.arange(len(key)), [x[1] for x in sort], tick_label = list(range(0, 32, 1)))
    name = titles[i].strip('./').strip('.64')
    plt.title(name)
    # plt.show()
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    fig.savefig(f'./histogram_{name}.png')
    fig.clf()

