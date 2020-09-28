from numpy import arange, zeros
from pandas import read_csv
from matplotlib.pyplot import bar

class myComputing(object):
    def __init__(self, file):
        self.file = file

    def to_raw_and_hist(self):
        title = self.file.split(self.file[:self.file.rfind('/') + 1])[1]
        img = read_csv(self.file)
        img = img.to_numpy().flatten()
        strings = ''

        for row in img:
            strings += row

        dic = {}
        for s in strings:
            if not s == '\x1a':
                dic[f'{str(s)}'] = strings.count(str(s))

        if img[-1] == '\x1a':
            new_img = zeros(shape = (img[:-1].shape[0], 64))
        else:
            new_img = zeros(shape = (img[:].shape[0], 64))

        for row in range(new_img.shape[0]):
            for col in range(new_img.shape[1]):
                last_row = -1 if img[-1] == '\x1a' else new_img.shape[1] + 1

                if img[:last_row][row][col].isalpha():
                    new_img[row][col] = ord(img[:last_row][row][col]) - 55
                else:
                    new_img[row][col] = img[:last_row][row][col]

        print(new_img.shape, new_img)
        
        # value = list(dic.values())
        key = list(dic.keys())
        sort = sorted(dic.items(), key = lambda x: x[0])

        self.key = arange(len(key))
        self.height = [x[1] for x in sort]
        self.title = title
        self.raw_img = new_img

    def add_processed(self, value):
        self.processed_img = self.raw_img + value

    def mul_processed(self, value):
        self.processed_img = self.raw_img * value
        