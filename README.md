Implementation of the music recommendation model described in [*Deep Temporal Neural Music Recommendation Model Utilizing Music and User Metadata*](https://www.mdpi.com/2076-3417/9/4/703). The database used in the paper is [WSDM-KKBOX](https://www.kaggle.com/c/kkboxmusic-recommendation-challenge/data).

The **DTNMR** model is based on extracting music characteristics and users’ intrinsic preferences from
their metadata. Given a user's metadata and the user's history of played songs, the objective is to provide the best song recommendation for that user to listen to.

---

#### Description of this repo

The current source code includes:
- data loading from .csv
- user and song processing
- preparation of data points for training and testing
- model definition in PyTorch
- model wrapper to perform training, validation, and testing

*records designate the association between a song and a user i.e. record (user_1, song_1) corresponds to 'user1 has listened to song song_1'.*

---

#### Some statistics on WSDM-KKBOX:

Initially, the database was used for a competition but the objectives were different.
The training set contains 7M records, the test set contains 2.5M records, with 2.2M unique songs and nearly 35k users. The number of songs listened to by users can go from one to several hundred. Due to the nature of the DTNMR model, and the fact that entities are encoded in multi-hot encodings, it was required to reduce the amount of data while still retaining quality in the datapoints. In particular, it seemed important to keep users which had enough songs in their playlist for the 'temporal' aspect to play in. Users with very few songs did not seem valuable for the training set. An obvious approach was to sort the songs by their popularity level [**currently based on the popularity in the training set!**].
The upper limit for the user and song encoding sizes was set to 10k bits.

After the pre-processing phase, the database subset contains a total of 11k unique users [**number of unique songs?**] and 240k points for training, and 38k points for testing [**check for duplicates**]. The user encoding and song encoding sizes were around 9,200 bits each.

---

#### Model structure

![DTNMR model structure](https://i.stack.imgur.com/dJEdv.png)

*A thousand words leave not the same deep impression as does a single deed.* - Henrik Ibsen

---

#### Usage

    python run.py [--reload]

On the first run, it will perform the initial data extraction and preprocessing. This will dump the processed data into four separate files `songs.p`, `users.p`, `training.p`, and `testing.p`. These files will be used to initialize the different encoders and torch Datasets.

Other possible arguments:
- `--reload`: this flag can be used to force refresh cache. This is especially useful when changing the different preprocessing parameters, see next [**TODO**].
- `--top NB`: will remove all songs which are not among the NB% most popular songs.
- `--min-len NB`: the minimum number of songs in a playlist for the user to be used for training.
- `--short-term-len NB`: the length of the short-term playlists.
- `--long-term-len NB`: the length of the long-term playlists, Must be lower than `--min-len`.

