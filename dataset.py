import random
from torch.utils.data import Dataset

class MRSDataset(Dataset):
    """
    User u has a playlist [s_1, s_2, s_3, ..., s_n] ordered in time. When picking an entry in
    the dataset, we randomly select `s_i` to be the label of the current task and consider each
    song previous to s_i to be known by the system. Therefore, p_i = [s_1, s_2, ..., s_i-1]
    will the playlist known to the system when predicting s_i to be the highest scoring song.
    """

    def __init__(self, user_song, playlist, users, songs, user_enc, song_enc, behavior_enc,
        subset_size, Lt_playlist_len):
        self.user_song = user_song
        self.playlist = playlist
        self.users = users
        self.songs = songs

        self.user_enc = user_enc
        self.song_enc = song_enc
        self.behavior_enc = behavior_enc
        self.subset_size = subset_size
        self.Lt_playlist_len = Lt_playlist_len

    def __getitem__(self, index):
        """
        user_song is an ordered list containing user-song pairs. It is the source of truth
        for picking training points:
            [0: (user, song), ..., index: (user_id, label_id), ..., n: (user, song)]
                                                    |
                                                   u, s
        From there, we locate the song's position in the user's complete playlist:
            [s_1, ...,  s_k  ...,  s_i-1,  s_i,  ...,  s_n]
                         \___________/      |
                       partial playlist  label=label_id

        This partial playlist will be used in two different ways: merged in the encoding of the
        user and encoded as a sequences of varying length which will be fed to a RNN.

        Returns
        ---
        Returned vectors are: (user, playlist, behaviors, subset)
            user: the user static encoding
            playlist: sequence of song encodings
            behaviors: sequence of behavior encodings
            subset: a sample list of song encodings including the label in 1st position
        """
        user_id, label_id = self.user_song[index]
        playlist = []
        behaviors = []
        for song_id, behavior in self.playlist[user_id]:
            if song_id == label_id:
                break
            playlist.append(song_id)
            behaviors.append(behavior)

        # Encode both the song's metadata and the user's behavior when listening to that song.
        for i, (song, behavior) in enumerate(zip(playlist, behaviors)):
            playlist[i] = self.encode_song(song)
            behaviors[i] = self.behavior_enc([behavior])

        user = self.encode_user(user_id, playlist)

        playlist_Lt = playlist[-self.Lt_playlist_len:]
        behaviors_Lt = behaviors[-self.Lt_playlist_len:]

        # Randomly sample a subset from the complete list of songs.
        subset_ids = random.sample(self.songs.keys(), self.subset_size - 1)
        subset = [self.encode_song(song_id) for song_id in subset_ids]

        label = self.encode_song(label_id)

        return user, playlist_Lt, behaviors_Lt, [label] + subset

    def encode_user(self, user_id, playlist):
        return self.user_enc(*self.users[user_id], playlist)

    def encode_song(self, song_id):
        return self.song_enc(*self.songs[song_id])

    def __len__(self):
        return len(self.user_song)
