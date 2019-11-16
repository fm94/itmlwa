# -*- coding: utf-8 -*-

#******************************************************************************
#
# Name        : extraction.py
# Description : extract 128 audio features, 1024 video features and labels
# Author      : Fares Meghdouri

#******************************************************************************

import google
import tensorflow
import json
import pandas as pd
import numpy as np

#******************************************************************************

N = 3844
#N = 100 # test dataset
chosen_labels = ['852', '1848']
output_file = '/localdisk/fm-youtube8m/train_full_{}.csv'.format('_'.join(chosen_labels))
input_dir = 'dataset/'

#******************************************************************************

def main():

    #l = []
    error_log = []

    with open(output_file, 'a') as f:
        for index in range(N):
            l = []
            try:
                for r in tensorflow.compat.v1.python_io.tf_record_iterator('{}'.format(input_dir)+'train%04d.tfrecord' % index):
                    a = tensorflow.train.Example.FromString(r)
                    #id = a.features.feature['id'].bytes_list.value[0].decode('utf-8')
                    b = google.protobuf.json_format.MessageToJson(a)
                    c = json.loads(b)
                    labels = c['features']['feature']['labels']['int64List']
                    if len(labels['value']) == 1 and labels['value'][0] in chosen_labels:
                        v = c['features']['feature']['mean_rgb']['floatList']['value']
                        a = c['features']['feature']['mean_audio']['floatList']['value']
                        l.append([v, a, labels['value'][0]])

                if len(l)!=0:
                    df = pd.DataFrame(l, columns=['rgb', 'audio', 'label'])
                    df1 = pd.DataFrame([{x: y for x, y in enumerate(item)} for item in df['audio'].values.tolist()], index=df.index)
                    df1 = df1.add_prefix('audio_')
                    df2 = pd.DataFrame([{x: y for x, y in enumerate(item)} for item in df['rgb'].values.tolist()], index=df.index)
                    df2 = df2.add_prefix('rgb_')
                    final = pd.concat([df1, df2, df['label']], axis=1)
                    if index == 0:                    
                        final.to_csv(f, header=True, index=False)
                    else:
                        final.to_csv(f, header=False, index=False)
                print('>> finished with {}'.format(input_dir)+'train%04d.npy' % index)

            except Exception as e:
                error_log.append(index)
                print('>> error {}'.format(e))

    print(len(error_log))


if __name__ == "__main__":
    main()
