An implementation of the music recommendation model described in [*Deep Temporal Neural Music Recommendation Model Utilizing Music and User Metadata*](https://www.mdpi.com/2076-3417/9/4/703). The database used is [WSDM-KKBOX](https://www.kaggle.com/c/kkbox-music-recommendation-challenge).

The **DTNMR** model is based on extracting music characteristics and users’ intrinsic preferences from
their metadata. Given a user's metadata and the user's history of played songs, the objective is to provide the best song recommendation for that user to listen to.

---

#### Description of this repo

The current source code includes:
- Data pre-processing;
- Preparation of data points for training and testing;
- Model definition in PyTorch;
- Model wrapper to perform training, validation, and testing.

---

#### Some statistics on WSDM-KKBOX:

Initially, the database was used for a competition but the objectives were different.
The training set contains 7M user-item interactions, the test set contains 2.5M. In total there are 2.2M unique songs and nearly 35k users. The number of songs listened to by users can go from one to several hundred. Due to the nature of the DTNMR model, and the fact that entities are encoded in multi-hot encodings, it was required to reduce the amount of data while still retaining quality in the datapoints. In particular, it seemed important to keep users which had enough songs in their playlist for the 'temporal' aspect to play in. Users with very few songs did not seem valuable for the training set. An obvious approach was to sort the songs by their popularity level.
The upper limit for the user and song encoding sizes was set to 10k bits.

After the pre-processing phase, the database subset contains a total of 11k unique users [**number of unique songs?**] and 240k points for training, and 38k points for testing [**check for duplicates**]. The user encoding and song encoding sizes were around 9,200 bits each.

---

#### Model structure

![DTNMR model structure](https://i.stack.imgur.com/bl1QF.png)

*A thousand words leave not the same deep impression as does a single deed.* - Henrik Ibsen

---

#### Usage

    python run.py [-h] [--reload] [--top RATIO] [--min-playlist-len NB] [--max-playlist-len NB] [--subset-size NB] [--short-term-len NB] [--long-term-len NB]

During the first run, it will perform the initial data extraction and preprocessing. This will dump the processed data into four separate files `songs.p`, `users.p`, `training.p`, and `testing.p`. These files will be used to initialize the different encoders and torch Datasets.

Optional arguments:

- `--reload `             Add flag to perform initial data extraction and pre-processing from .csv files.
- `--top` RATIO           Will only keep the songs among the RATIO in popularity.
- `--min-playlist-len` NB Minimum admissible playlist size, all users with a lower number will be left out.
- `--max-playlist-len` NB Maximum admissible playlist size, all users with a higher number will be left out.
- `--db` PATH             Directory where the WSDM-KKBOX csv files are located.
- `--cache` PATH          Directory where the processed data is or will be saved.
- `--subset-size` NB      Number of items in the negative sampling.
- `--short-term-len` NB   Length of a short-term user playlist.
- `--long-term-len` NB    Length of a long-term user playlist.


---

#### Reference

[1] Zheng, H.-T.; Chen, J.-Y.; Liang, N.; Sangaiah, A.K.; Jiang, Y.; Zhao, C.-Z. A Deep Temporal Neural Music Recommendation Model Utilizing Music and User Metadata. Appl. Sci. 2019, 9, 703

---

#### TODO

- Model evaluation (+ add weight/optim checkpoints) support for arguments and training parameters.
- Add support for model params in CLI.
