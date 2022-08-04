# -*- coding:  UTF-8 -*-
from __future__ import division
import math
import util
import time
import pandas as pd
import numpy as np
import datetime
import geohash
import random
import string
import os
from pathlib import Path


dat_suffix = ['toro', 'edin', 'glas', 'melb', 'osaka', 'weeplaces_poi_1000_length_9', 'weeplaces_poi_101_length_9']
dat_suffix = ['weeplaces_poi_10000_length_-1']
pwd = os.getcwd()
output_dir = f'{pwd}/output/'
Path(output_dir).mkdir(parents = True, exist_ok = True)
for filename in dat_suffix:
    precision = 6
    poi_path = f'./origin_data/poi-{filename}.csv'
    traj_path = f'./origin_data/traj-{filename}.csv'
    op_tdata = open(poi_path, 'r')
    points = util.read_POIs(op_tdata)
    final_dir = f'{output_dir}/{filename}'
    Path(final_dir).mkdir(parents = True, exist_ok = True)
    df = pd.read_csv(traj_path)
    poi_df = pd.read_csv(poi_path)
    poi_id_set = poi_df['poiID'].unique()
    poi_transition_matrix = {}
    for poi_id in poi_id_set:
        poi_transition_matrix[poi_id] = {}

    traj_df = pd.read_csv(traj_path)
    print('check-in', len(traj_df['trajLen'].index))
    print('trajectories', len(traj_df['trajID'].unique()))
    print(traj_df['trajLen'].mean())
    traj_df = traj_df[traj_df['trajLen'] >= 3]
    last_traj_id = -1
    last_poi = -1
    last_end_time = -1
    print(traj_df)
    for index, row in traj_df.iterrows():
        trajID = row['trajID']
        poi = row['poiID']
        start_time = row['startTime']
        if trajID == last_traj_id:
            if poi not in poi_transition_matrix[last_poi]:
                poi_transition_matrix[last_poi][poi] = 0
            poi_transition_matrix[last_poi][poi] += 1

            #print('from', last_poi, 'to', poi, 'timetothere', (start_time - last_end_time))
        last_poi = poi
        last_traj_id = trajID
        last_end_time = row['endTime']
    #random.choices(a_list, distribution)
    for key, value in poi_transition_matrix.items():
        count = 0
        keys = list(poi_transition_matrix[key].keys())
        for sub_key in keys:
            count += poi_transition_matrix[key][sub_key]
        p = []
        for sub_key in keys:
            poi_transition_matrix[key][sub_key] /= count
            p.append(poi_transition_matrix[key][sub_key])
        poi_transition_matrix[key] = {'p': p, 'poi': keys}
    # 3, 4, 5, 6
    print(filename, )
    index = traj_df['trajLen'].value_counts().index
    values = traj_df['trajLen'].value_counts().values
    t = (values / index)
    for i in range(len(index)):
        print('length ', index[i], ':', values[i]/ index[i])


    max_length = np.max(traj_df['trajLen'])
    expected_size = 50000
    length_probability = values/values.sum(axis=0,keepdims=1)
    currentSize = 0
    trajectories = {}
    while currentSize < expected_size:
        last_poi = poi_id_set[np.random.choice(len(poi_id_set), 1)[0]]
        length_choice_index = np.random.choice(len(length_probability), 1, p = length_probability)[0]
        fake_traj_length = index[length_choice_index]
        temp = [[str(last_poi), '[MASK]', '[MASK]']]
        while len(temp) < fake_traj_length:
            t_matrix = poi_transition_matrix[last_poi]
            poi_list = t_matrix['poi']
            p = t_matrix['p']
            if len(t_matrix['poi']) > 0:
                choice_index = np.random.choice(len(poi_list), 1, p = p)[0]
                next_poi_id = poi_list[choice_index]
                temp.append([str(next_poi_id), '[MASK]', '[MASK]'])
                last_poi = next_poi_id
            else:
                break
        if len(temp) == fake_traj_length:
            user_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
            key = f'{user_id}-{currentSize}'
            trajectories[key] = temp
            currentSize += 1
        if currentSize % 1000 == 0:
            print(filename, " currentSize: ", currentSize)
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
    'hour': set(),
    }
    poi_ghash = {}
    poi_cat = {}
    max_length = 0
    for key in trajectories:
        trajectory_list = trajectories[key]
        matches = key.split('-')
        user_id = matches[0]
        traj_id = matches[1]
        input_place_id = []
        sequence_place_id = []
        sequence_category = []
        sequence_geohash = []
        sequence_week = []
        sequence_day = []
        sequence_hour = []
        if len(trajectory_list) > max_length:
            max_length = len(trajectory_list)
        for trajectory in trajectory_list:
            place_id = trajectory[0]
            timestamp = '[MASK]'
            week = '[MASK]'
            day = '[MASK]'
            hour = '[MASK]'
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
            unique['hour'].add(hour)
            poi_ghash[place_id] = ghash
            poi_cat[place_id] = cat

            sequence_category.append(cat)
            sequence_geohash.append(ghash)
            sequence_place_id.append(place_id)
            sequence_week.append(week)
            sequence_day.append(day)
            sequence_hour.append(hour)
        sequence_place_id = ','.join(sequence_place_id)
        sequence_geohash = ','.join(sequence_geohash)
        sequence_category = ','.join(sequence_category)
        sequence_week = ','.join(sequence_week)
        sequence_day = ','.join(sequence_day)
        sequence_hour = ','.join(sequence_hour)
        checkins.append([user_id, sequence_place_id, sequence_geohash, sequence_category, sequence_week, sequence_day, sequence_hour])
    print(f'{filename}: {max_length}')
    df = pd.DataFrame(checkins)
    #random_selection = np.random.rand(len(df.index)) <= 0.85
    #train_data = df[random_selection]
    #test_data = df[~random_selection]

    df.to_csv(f"{final_dir}/pretrain_data.csv", index = False, sep = "|", header = False)
    #df2 = pd.DataFrame.from_dict(poi_ghash, orient='index')
    #df2.to_csv(f"{final_dir}/pretrain_poi_ghash.csv", index = True, sep = "|", header = False)
    #df3 = pd.DataFrame.from_dict(poi_cat, orient='index')
    #df3.to_csv(f"{final_dir}/pretrain_poi_cat.csv", index = True, sep = "|", header = False)

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
