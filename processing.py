import os, pickle
from utils.process import process, no_special_character

def process_training(train):
    popularity = {}
    most_popular = 0
    for user, song in zip(train['msno'], train['song_id']):
        popularity[song] = popularity[song]+1 if song in popularity else 1
        if (popularity[song] > most_popular):
            most_popular = popularity[song]
    return popularity, most_popular


def process_songs(songs, popularity, most_popular, top):
    print('processing songs...')

    indices = {x: i for i, x in enumerate(songs['song_id'])}
    ids = songs['song_id']

    def process_feature(data, transform=lambda x: x, policy=lambda x: True):
        return process(data, songs['song_id'], indices, transform, policy)

    # Tag songs by going through the feature values one feature at a time.
    _, __ = process_feature(songs['song_id'],
        transform=lambda x: popularity[x]/most_popular if x in popularity else 0,
        policy=lambda x: x > top)
    length, length_set = process_feature(songs['song_len'],
        transform=lambda x: int(x)/1000, policy=lambda x: x < 360)
    genre, genre_set = process_feature(songs['genre_ids'])
    artist, artist_set = process_feature(songs['artist_name'],
        transform=lambda x: x.strip(), policy=no_special_character)
    composer, composer_set = process_feature(songs['composer'],
        transform=lambda x: x.strip(), policy=no_special_character)
    language, language_set = process_feature(songs['language'])

    songs = {id: data for id, *data in zip(ids, length, genre, artist, composer, language)
        if indices[id] != -1}
    return songs, {max(length_set)}, genre_set, artist_set, composer_set, language_set


def process_users(users):
    print('processing users...')

    indices = {x: i for i, x in enumerate(users['msno'])}

    def process_feature(data, transform=lambda x: x, policy=lambda x: True):
        return process(data, users['msno'], indices, transform, policy)

    age, age_set = process_feature(users['bd'], policy=lambda x: 15 < int(x) and int(x) < 60)
    gender, gender_set = process_feature(users['gender'])
    city, city_set = process_feature(users['city'])

    users = {id: data for id, *data in zip(users['msno'], age, gender, city) if indices[id] != -1}
    return users, age_set, gender_set, city_set


def construct_datapoints(data, users, songs, min_len, max_len):
    print('constructing set of records...')

    # Construct the playlist dictionnary {user_id: (song_id[], behavior)} ; behavior set.
    playlist, behaviors = {}, set()
    for user, song, behavior in zip(data['msno'], data['song_id'], data['source_system_tab']):
        if user in users and song in songs:
            behaviors.add(behavior)
            playlist.setdefault(user, []).append((song, behavior))

    # Construct the list of training points (user, song).
    points = []
    for user, p in playlist.items():
        if (min_len < len(p) < max_len):
            # Each data point should have at least min_len songs before it in the user's playlist.
            points += [(user, song) for song, _ in p[min_len:]]
    return points, playlist, behaviors
