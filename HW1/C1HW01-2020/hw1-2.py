import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from IPython import embed

def to_hist(array):
    dic = {}
    for row in range(array.shape[0]):
        for col in range(array.shape[1]):

            if int(array[row][col]) in dic:
                if int(array[row][col]) >= 0 and int(array[row][col]) < 32:
                    dic[int(array[row][col])] += 1

                elif int(array[row][col]) < 0:
                    dic[0] += 1

                else:
                    dic[31] += 1

            else:
                if int(array[row][col]) >= 0 and int(array[row][col]) < 32:
                    dic[int(array[row][col])] = 1

                elif int(array[row][col]) < 0:
                    if 0 in dic:
                        dic[0] += 1
                    else:
                        dic[0] = 1
                else:
                    if 31 in dic:
                        dic[31] += 1
                    else:
                        dic[31] = 1
         
    for k in range(0, 32, 1):
        if k not in dic:
            dic[k] = 0

    dic = sorted(dic.items(), key = lambda x: x[0])

    return dic

imgs = []
strings = ['', '', '', '']
sets = []
titles = []
new_imgs = []
for img in glob('./*.64'):
    titles.append(img)
    x = pd.read_csv(img)
    imgs.append(x.values.flatten())

for img in range(len(imgs)):
    if img == 1:
        break

    if imgs[img][-1] == '\x1a':
        new_img = np.zeros(shape = (imgs[img][:-1].shape[0], 64))
    else:
        new_img = np.zeros(shape = (imgs[img][:].shape[0], 64))

    for row in range(new_img.shape[0]):
        for col in range(new_img.shape[1]):
            last_row = -1 if imgs[img][-1] == '\x1a' else new_img.shape[1] + 1

            if imgs[img][:last_row][row][col].isalpha():
                new_img[row][col] = ord(imgs[img][:last_row][row][col]) - 55
            else:
                new_img[row][col] = imgs[img][:last_row][row][col]
    # new_img *= 8
    # new_img -= 2
    # new_img *= 1.5

    new_imgs.append(new_img)

    print(new_img)
    dic = to_hist(new_img)

    print(dic)
    print(titles[img])
    # plt.bar(list(range(len(dic))), [x[1] for x in dic], tick_label = [x[0] for x in dic])
    # name = titles[img].strip('./').strip('.64')
    # plt.title("mult1.5" + name)

    # fig = plt.gcf()
    # fig.set_size_inches(15, 10)
    # fig.savefig(f'./mult1.5_histogram_{name}.png')
    # plt.imshow(new_img, cmap = 'gray', vmin = 0, vmax = 255)
    # plt.show()

shift_img = np.zeros(shape = new_imgs[0].shape)
for row in range(new_imgs[0].shape[0]):
    for col in range(new_imgs[0].shape[1]):

        if col > 0:
            shift_img[row][col - 1] = new_imgs[0][row][col]
        else:
            shift_img[row][col] = 0

# print(new_imgs[0] - shift_img)
# print(to_hist(new_imgs[0] - shift_img))

# plt.imshow(new_imgs[0] - shift_img, cmap = 'gray')
# plt.show()
# avg_img = to_hist(((new_imgs[0] + new_imgs[1]) / 2).astype(int))
# plt.bar(list(range(len(dic))), [x[1] for x in dic], tick_label = [x[0] for x in dic])
# name = 'Average of LINCOLN and LISA'
# plt.title(name)

# fig = plt.gcf()
# fig.set_size_inches(15, 10)
# fig.savefig(f'./{name}.png')
# plt.show()

