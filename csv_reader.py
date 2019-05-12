import pandas
import random
import json

random.seed(233)


def save_data(path, data_list):
    json_list = []
    for stc, label in data_list:
        sample = {}
        sample['question'] = stc
        sample['label'] = label
        json_list.append(json.dumps(sample))
    json_list = '\n'.join(json_list)
    with open(path, 'w', encoding='utf-8') as w:
        w.write(json_list)


datas = pandas.read_csv('./test.csv')
label_list = datas['label'].tolist()
stc_list = datas['question'].tolist()
c = list(zip(stc_list, label_list))
random.shuffle(c)
stc_list[:], label_list[:] = zip(*c)
train_len = int(0.75 * len(stc_list))
test_len = int(0.9 * len(stc_list))
train_data = c[:train_len]
valid_data = c[train_len:test_len]
test_data = c[test_len:]
save_data('data/train.json', train_data)
save_data('data/valid.json', valid_data)
save_data('data/test.json', test_data)

