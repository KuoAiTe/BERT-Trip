# -*- coding:  UTF-8 -*-
from __future__ import division
import math
import util
import time
import pandas as pd
import numpy as np
import datetime
import geohash
import os
from pathlib import Path


dat_suffix = [ 'edin', 'glas', 'melb', 'osaka', 'toro']
pwd = os.getcwd()
output_dir = f'{pwd}/output/'
Path(output_dir).mkdir(parents = True, exist_ok = True)
for filename in dat_suffix:
    precision = 6
    poi_name = "poi-" + filename + ".csv"
    tra_name = "traj-" + filename + ".csv"
    op_tdata = open('./origin_data/' + poi_name, 'r')
    ot_tdata = open('./origin_data/' + tra_name, 'r')
    final_dir = f'{output_dir}/{filename}'
    Path(final_dir).mkdir(parents = True, exist_ok = True)
    #print('To Train',dat_suffix[dat_ix])
    def create_vocab_file(unique):
        for key in unique:
            prepend = ['[MASK]','[CLS]','[PAD]','[SEP]','[UNK]']
            data = prepend + list(unique[key])
            d = pd.DataFrame.from_dict(data)
            d.to_csv(f'{final_dir}/{key}_vocab.txt', index = False, header = None)

    points = util.read_POIs(op_tdata)
    (trajectories, users, poi_count) = util.read_trajectory(ot_tdata)
    trajectories = util.filter_trajectory(trajectories)
    checkins = []
    max_length = 0
    for key in trajectories:
        trajectory_list = trajectories[key]
        length = len(trajectory_list)
        if length > max_length:
            max_length = length
    padding = 0

    unique = {
    'poi': set(),
    'user': set(),
    'cat': set(),
    'ghash': set(),
    'week': set(),
    'day': set(),
    'quarter_hour': set(),
    }
    poi_ghash = {}
    poi_cat = {}
    max_length = 0
    incorrect_trajectory = 0
    for key in trajectories:
        trajectory_list = trajectories[key]
        matches = key.split('-')
        user_id = matches[0]
        traj_id = matches[1]
        input_place_id = []
        data = []
        if len(trajectory_list) > max_length:
            max_length = len(trajectory_list)
        for trajectory in trajectory_list:
            place_id = trajectory[0]
            timestamp = int(trajectory[2])
            dt_object = datetime.datetime.fromtimestamp(timestamp)
            week = str(dt_object.isocalendar()[1])
            day = str(dt_object.isocalendar()[2])
            quarter_hour = str(int(dt_object.hour) * 4 + int(dt_object.minute / 15))

            cat = points[place_id]['cat'].replace(' ', '').replace(',', '-')
            lat = points[place_id]['lat']
            lon = points[place_id]['lon']
            ghash = geohash.encode(float(lat), float(lon), precision = precision)
            unique['poi'].add(place_id)
            unique['user'].add(user_id)
            unique['cat'].add(cat)
            unique['ghash'].add(ghash)
            unique['week'].add(week)
            unique['day'].add(day)
            unique['quarter_hour'].add(quarter_hour)
            poi_ghash[place_id] = ghash
            poi_cat[place_id] = cat
            timestamp = str(timestamp)
            data.append({'place_id': place_id, 'ghash': ghash, 'cat': cat, 'week': week, 'day': day, 'quarter_hour': quarter_hour, 'timestamp': timestamp})
        timestamps = list(map(lambda x: int(x['timestamp']), data))
        last_timestamp = -1
        for timestamp in timestamps:
            if last_timestamp == -1:
                last_timestamp = timestamp
            else:
                if last_timestamp > timestamp:
                    incorrect_trajectory += 1
                    break
        #data.sort(key = lambda x: x['timestamp'])

        sequence_place_id = ','.join(map(lambda x: x['place_id'], data))
        sequence_geohash = ','.join(map(lambda x: x['ghash'], data))
        sequence_category = ','.join(map(lambda x: x['cat'], data))
        sequence_week = ','.join(map(lambda x: x['week'], data))
        sequence_day = ','.join(map(lambda x: x['day'], data))
        sequence_quarter_hour = ','.join(map(lambda x: x['quarter_hour'], data))
        sequence_timestamp = ','.join(map(lambda x: x['timestamp'], data))
        checkins.append([user_id, sequence_place_id, sequence_geohash, sequence_category, sequence_week, sequence_day, sequence_quarter_hour, sequence_timestamp])
    print(f'{filename}: {max_length} incorrect_trajectory: {incorrect_trajectory} / {len(trajectories)} = {incorrect_trajectory / len(trajectories)}')
    #create_vocab_file(unique)
    df = pd.DataFrame(checkins)
    #print(df)
    df.to_csv(f"{final_dir}/data.csv", index = False, sep = "|", header = False)
    #df2 = pd.DataFrame.from_dict(poi_ghash, orient='index')
    #df2.to_csv(f"{final_dir}/poi_ghash.csv", index = True, sep = "|", header = False)
    #df3 = pd.DataFrame.from_dict(poi_cat, orient='index')
    #df3.to_csv(f"{final_dir}/poi_cat.csv", index = True, sep = "|", header = False)

#print(trajectories)
#print(trajectories)
#distance_count = util.disc_count(points, trajectories)
#upper_dis = max(distance_count)

#train_trajectories, train_users, train_time, train_distances = util.generate_train_data(points, trajectories, upper_dis)
#print(poi_count)
#print('poi number is',len(poi_count) - 3)
#voc_poi = [poi_id for poi_id, count in poi_count]
#int_to_vocab, vocab_to_int = util.extract_words_vocab(voc_poi)
#print(int_to_vocab)
#print(vocab_to_int)

#generate traning dataset
#train_dataset = util.generate_train_dataset(vocab_to_int, train_trajectories)
#util.write_train_dataset(embedding_name, train_dataset, train_users)
