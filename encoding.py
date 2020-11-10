import numpy as np
from utils.encoders import Encoder, LinearEncoder, MHEncoder

class UserEncoder(Encoder):
    """User encoder: encodes the user's age, gender, city and complete playlist."""

    def __init__(self, ages, genders, cities, song_encoder):
        assert(isinstance(song_encoder, SongEncoder))
        super().__init__(MHEncoder(ages), MHEncoder(genders), MHEncoder(cities))
        self.song_encoder = song_encoder

    def playlist_signature(self, playlist_encoded):
        return np.clip(sum(playlist_encoded), 0, 1)[1:] # remove linear component i.e. here the 1st

    def __call__(self, age, gender, city, playlist_encoded):
        user_encoded = super().__call__(age, gender, city)
        playlist_signature = self.playlist_signature(playlist_encoded)
        return np.concatenate([user_encoded, playlist_signature])

    def __len__(self):
        return super().__len__() + len(self.song_encoder) - 1


class SongEncoder(Encoder):
    """Song encoder: encodes the song's length, genres, artists, composers and languages."""

    def __init__(self, lengths, genres, artists, composers, languages):
        super().__init__(LinearEncoder(lengths), MHEncoder(genres), MHEncoder(artists),
            MHEncoder(composers), MHEncoder(languages))


class BehaviorEncoder(Encoder):
    """
    Song listening behavior encoding: a wrapper for a single MHEncoder. It's needed to
    manually merge some feature values together e.g. 'null' and 'settings'.
    """

    def __init__(self, behaviors):
        behaviors.discard('null')
        behaviors.discard('settings')
        super().__init__(MHEncoder(behaviors))

    def __call__(self, behavior):
        if (behavior == 'null' or behavior == 'settings'):
            behavior = ''
        return super().__call__(behavior)
