from statistics import mean

import torch
import torch.nn as nn
from torch.nn import Module, Linear
from tqdm import tqdm

from utils.nn import MLP, RNN

class DTNMRWrapper:
    """Runner wrapper for the DTNMR model."""

    def __init__(self, model, train_dl, valid_dl, optimizer):
        """
        Parameters
        ---
            model: torch.nn.Model, The recommendation model to train.
            train_dl: torch.utils.data.DataLoader, the training data loader
            valid_dl: torch.utils.data.DataLoader, the validation data loader
            optimizer: torch.optim, the optimizer used for back-propagation
        """
        self.model = model
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epochs):
        for epoch in range(epochs):
            print('Epoch %i/%i' % (epoch+1, epochs))
            losses, accuracies = self.train_epoch()
            print('summary of epoch: average loss={:.2f}, average accuracy={:.2f}' \
                .format(mean(losses), mean(accuracies)))

    def train_epoch(self):
        """Runs the training on the whole training set."""
        losses, accuracies = [], []
        for i, x in (t := tqdm(enumerate(self.train_dl), total=len(self.train_dl))):
            y = torch.zeros((x[0].shape[0],1), dtype=torch.long) #? add , 1) if necessary
            y_hat = self.model(*x)

            self.optimizer.zero_grad()
            loss = self.criterion(y_hat, y)
            loss.backward()
            self.optimizer.step()

            loss = loss.item()
            accuracy = (torch.argmax(y_hat, dim=1) == 0).float().mean().item()
            losses.append(loss)
            accuracies.append(accuracy)
            t.set_description("loss %.2f - accuracy %.2f%%" % (loss, accuracy*100))
        return losses, accuracies


class DTNMR(Module):
    """
    The recommendation neural network is comprised of 4 components.
        [USFC] User Static Feature Component
            USFC(user_encoding) = user_static_embedding
        [MFC] Music Feature Component:
            song_embedding = MFC(song_encoding)
        [UDFC] User Dynamic Feature Component: long-term & short-term
            song_playlist_embedding = UDFC(song_embedding[])
        [RC] Rating Component
            rating = RC(user_embedding, song_embedding)
    """

    def __init__(self, user_mhe_size, song_mhe_size, behavior_mhe_size, st_playlist_len, emb_size):
        """Given the entity encoding sizes, it will construct the different components."""
        super(DTNMR, self).__init__()
        self.user_mhe_size = user_mhe_size
        self.song_mhe_size = song_mhe_size
        self.emb_size = emb_size
        self.song_emb_size = emb_size + behavior_mhe_size
        self.st_playlist_len = st_playlist_len

        self.USFC = MLP(self.user_mhe_size, n_1=512, n_2=64, n_out=emb_size)
        self.MFC = MLP(self.song_mhe_size, n_1=512, n_2=64, n_out=emb_size)
        self.UDFC = RNN(self.song_emb_size, num_layers=2, n_hidden=256, n_out=emb_size)
        self.RC = Linear(2*emb_size, 1)

    def forward(self, user, playlist, behaviors, subset):
        """
        The forward pass of the model is the calculation of the rating between the user and
        each song in the subset. The result is the list [Score(u, s) for s in subset].
        """
        # User static feature embedding.
        user_static = self.USFC(user.float())

        # User dynamic feature embedding.
        latent_playlist = torch.stack([self.MFC(s.float()) for s in playlist])
        behaviors = torch.stack([b.float() for b in behaviors])
        Lt_playlist = torch.cat([latent_playlist, behaviors], dim=2)
        st_playlist = Lt_playlist[-self.st_playlist_len:]
        user_dynamic = self.UDFC(Lt_playlist) + self.UDFC(st_playlist)

        # User feature combined embedding.
        user_feature = user_static + user_dynamic

        # Rating evaluation on song subset.
        ratings = []
        for song in subset:
            embedding = self.MFC(song.float())
            combined = torch.cat([user_feature, embedding], dim=1)
            ratings.append(self.RC(combined))

        return torch.stack(ratings, dim=1)
