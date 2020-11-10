import csv
from collections import Counter

from tqdm import tqdm
import matplotlib.pyplot as plt

def histogram(x, title='', xlabel='Features', ylabel='Frequency'):
    freq = Counter(sorted(x))
    plt.figure(figsize=(18,8))
    plt.bar(freq.keys(), freq.values(), align='center', color='#3c3c3c', alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def get(fp):
    """Basic csv loader which separates the columns (i.e. features) into a dict."""
    with open(fp, newline='', encoding='utf8') as file:
        total = sum(1 for line in open(fp, encoding='utf8'))
        reader = csv.reader(file, delimiter=',')
        headers = next(reader, None)
        data = {h: [] for h in headers}
        print('Loading %s' % fp)
        for row in tqdm(reader, total=total):
            for h, v in zip(headers, row):
                data[h].append(v)
    return headers, data
