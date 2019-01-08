import numpy as np
import pandas as pd
import shutil, os, json

# img dir path
dataset_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(os.path.dirname(os.path.dirname(dataset_dir)), 'dataset')
img_root = os.path.join(dataset_dir, 'raw_imgs')

map_label = {
    'happiness': 0,
    'disgust': 1,
    'repression': 2,
    'surprise': 3,
    'others': 4
}

def select_adjacent_apex_frame(record, adjacent_frames_num=10):
    base_path = record['path']
    if record['ApexFrame'] != '/':
        ans = [os.path.join(base_path, 'reg_img{}.jpg'.format(x)) \
                for x in range(int(record['ApexFrame'])-int(adjacent_frames_num/2), int(record['ApexFrame'])+int(adjacent_frames_num/2))\
                if x >= int(record['OnsetFrame']) and x <= int(record['OffsetFrame'])]
    else:
        mid = int((int(record['OnsetFrame']) + int(record['OffsetFrame']) + 1) / 2)
        ans = [os.path.join(base_path, 'reg_img{}.jpg'.format(x)) \
                for x in range(mid-int(adjacent_frames_num/2), mid+int(adjacent_frames_num/2))\
                if x >= int(record['OnsetFrame']) and x <= int(record['OffsetFrame'])]
    return ans

def pd2dict(csv_name):
    data = process_csv_data(csv_name)
    print(data.columns)
    ans = []
    for i in range(1, data.shape[0]+1):
        tmp = {}
        tmp['label'] = data.loc[i, 'Estimated Emotion']
        tmp['path'] = 'sub' + data.loc[i, 'Subject'] + '/' + data.loc[i, 'Filename']
        tmp['OnsetFrame'] = data.loc[i, 'OnsetFrame']
        tmp['ApexFrame'] = data.loc[i, 'ApexFrame']
        tmp['OffsetFrame'] = data.loc[i, 'OffsetFrame']
        ans.append(tmp)
    print(ans)
    json.dump(ans, open(os.path.join('auxiliary', 'baseinfo.json'), 'w'), indent=4)
    return ans

def process_csv_data(csv_name):
    path = os.path.join(dataset_dir, csv_name)
    df=pd.read_csv(path, header=None, sep=',')
    _df = df.loc[1:]
    _df.columns = df.loc[0].values
    return _df

def make_img_dataset(json_file, num=10, target_dir='imgs'):
    _label = []
    data = json.load(open(json_file))
    for d in data:
        if not any(x == d['label'] for x in map_label.keys()):
            continue
        imgs = select_adjacent_apex_frame(d, adjacent_frames_num=num)
        for img in imgs:
            tmp = {
                'path': img,
                'label': map_label[d['label']]
            }
            _label.append(tmp)
    # print(_label)
    print(len(_label))
    json.dump(_label, open(os.path.join('auxiliary', 'single_img.json'), 'w'), indent=4)


# make_img_dataset('auxiliary/baseinfo.json')
