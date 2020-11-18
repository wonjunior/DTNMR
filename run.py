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

    # ~ data processing
    parser.add_argument('--reload', action='store_true', default=False,
        help='Add flag to perform initial data extraction and pre-processing from .csv files.')
    parser.add_argument('--top', metavar='RATIO', type=float, default=0.001,
        help='Will only keep the songs among the RATIO in popularity.')
    parser.add_argument('--min-playlist-len', metavar='NB', type=int, default=20,
        help='Minimum admissible playlist size, all users with a lower number will be left out.')
    parser.add_argument('--max-playlist-len', metavar='NB', type=int, default=200,
        help='Maximum admissible playlist size, all users with a higher number will be left out.')
    parser.add_argument('--db', metavar='PATH', type=str, default='../db',
        help='Directory where the WSDM-KKBOX csv files are located.')
    parser.add_argument('--cache', metavar='PATH', type=str, default='data',
        help='Directory where the processed data is or will be saved.')

    # ~ training params
    parser.add_argument('--subset-size', metavar='NB', type=int, default=5,
        help='Number of items in the negative sampling.')
    parser.add_argument('--short-term-len', metavar='NB', type=int, default=10,
        help='Length of a short-term user playlist.')
    parser.add_argument('--long-term-len', metavar='NB', type=int, default=20,
        help='Length of a long-term user playlist.')
    # bs, epochs, lr

    args = parser.parse_args()

    assert(os.path.isdir(args.db))

    ## Initial load, processing and saving
    if args.reload or not os.path.isdir('data'):
        os.makedirs('data', exist_ok=True)

        # Training data
        train_headers, train = get(os.sep.join((args.db, 'train.csv')))
        popularity, most_popular = process_training(train)

        # Songs
        song_headers, songs_csv = get(os.sep.join((args.db, 'songs.csv')))
        songs, *song_sets = process_songs(songs_csv, popularity, most_popular, top=args.top)
        save('data/songs.p', songs, *song_sets)

        # Users
        user_headers, users_csv = get(os.sep.join((args.db, 'members.csv')))
        users, *user_sets = process_users(users_csv)
        save('data/users.p', users, *user_sets)

        # Actual data train/test data points
        test_headers, test = get(os.sep.join((args.db, 'test.csv')))
        save('data/training.p', *construct_datapoints(train, users, songs,
            min_len=args.min_playlist_len, max_len=args.max_playlist_len))
        save('data/testing.p', *construct_datapoints(test, users, songs,
            min_len=args.min_playlist_len, max_len=args.max_playlist_len))


    ## Loading processed data and encoding
    # Songs
    songs, *song_sets = load(os.sep.join((args.cache, 'songs.p')))
    encode_song = SongEncoder(*song_sets)
    song_mhe_size = len(encode_song)

    # Users
    users, *user_sets = load(os.sep.join((args.cache, 'users.p')))
    encode_user = UserEncoder(*user_sets, encode_song)
    user_mhe_size = len(encode_user)

    # Playlists
    train_points, train_playlist, behaviors = load(os.sep.join((args.cache, 'training.p')))
    test_points, test_playlist, _ = load(os.sep.join((args.cache, 'testing.p')))

    # Behaviors
    encode_behavior = BehaviorEncoder(behaviors)
    behavior_mhe_size = len(encode_behavior)

    print('song_mhe_size= {},\nuser_mhe_size= {},\nbehavior_mhe_size= {}' \
        .format(song_mhe_size, user_mhe_size, behavior_mhe_size))
    print('train set: [{} users, {} interactions] - test set: [{} users, {} interactions]' \
        .format(len(train_playlist), len(train_points), len(test_playlist), len(test_points)))

    ## Training
    train_set = MRSDataset(train_points, train_playlist, users, songs, encode_user, encode_song,
        encode_behavior, subset_size=args.subset_size, Lt_playlist_len=args.long_term_len)
    train_dl = DataLoader(train_set, batch_size=16, shuffle=True)

    valid_set = MRSDataset(test_points, test_playlist, users, songs, encode_user, encode_song,
        encode_behavior, subset_size=args.subset_size, Lt_playlist_len=args.long_term_len)
    valid_dl = DataLoader(valid_set, batch_size=16, shuffle=True)


    model = DTNMR(user_mhe_size, song_mhe_size, behavior_mhe_size,
        st_playlist_len=args.short_term_len, emb_size=32)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    system = DTNMRWrapper(model, train_dl, valid_dl, optimizer)

    system.train(epochs=1)
