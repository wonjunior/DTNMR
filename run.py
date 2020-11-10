#!/usr/bin/env python
# coding: utf-8

import os, pdb
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from utils.misc import get
from utils.dumper import save, load
from processing import process_training, process_songs, process_users, construct_datapoints
from encoding import UserEncoder, SongEncoder, BehaviorEncoder
from dataset import MRSDataset
from model import DTNMR, DTNMRWrapper

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-r', '--reload', action='store_true', default=False,
        help='Add flag to perform initial data extraction and pre-processing from .csv files.')

    args = parser.parse_args()

    ## Initial load, processing and saving
    if args.reload or not os.path.isdir('data'):
        os.makedirs('data', exist_ok=True)

        # Training data
        train_headers, train = get('../db/train.csv')
        popularity, most_popular = process_training(train)

        # Songs
        song_headers, songs_csv = get('../db/songs.csv')
        songs, *song_sets = process_songs(songs_csv, popularity, most_popular)
        save('data/songs.p', songs, *song_sets)

        # Users
        user_headers, users_csv = get('../db/members.csv')
        users, *user_sets = process_users(users_csv)
        save('data/users.p', users, *user_sets)

        # Actual data train/test data points
        test_headers, test = get('../db/test.csv')
        save('data/training.p', *construct_datapoints(train, users, songs, min_len=20, max_len=200))
        save('data/testing.p', *construct_datapoints(test, users, songs, min_len=20, max_len=200))


    ## Loading processed data and encoding
    # Songs
    songs, *song_sets = load('data/songs.p')
    encode_song = SongEncoder(*song_sets)
    song_mhe_size = len(encode_song)

    # Users
    users, *user_sets = load('data/users.p')
    encode_user = UserEncoder(*user_sets, encode_song)
    user_mhe_size = len(encode_user)

    # Playlists
    train_points, train_playlist, behaviors = load('data/training.p')
    test_points, test_playlist, _ = load('data/testing.p')

    # Behaviors
    encode_behavior = BehaviorEncoder(behaviors)
    behavior_mhe_size = len(encode_behavior)

    print('song_mhe_size= %i,\nuser_mhe_size= %i,\nbehavior_mhe_size= %i,\nnb unique users in train set= %i,\nnb unique users in test set= %i,\nnb points for training= %i\nnb points for testing= %i' % \
        (song_mhe_size, user_mhe_size, behavior_mhe_size, len(train_playlist), len(test_playlist), len(train_points), len(test_points)))


    ## Training
    train_set = MRSDataset(train_points, train_playlist, users, songs, encode_user, encode_song,
        encode_behavior, subset_size=5, LT_playlist_len=20)
    train_dl = DataLoader(train_set, batch_size=16, shuffle=True)

    valid_set = MRSDataset(test_points, test_playlist, users, songs, encode_user, encode_song,
        encode_behavior, subset_size=5, LT_playlist_len=20)
    valid_dl = DataLoader(valid_set, batch_size=16, shuffle=True)


    model = DTNMR(user_mhe_size, song_mhe_size, behavior_mhe_size, ST_playlist_len=10, emb_size=32)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    system = DTNMRWrapper(model, train_dl, valid_dl, optimizer)

    system.train(epochs=1)
