import numpy as np

class LinearEncoder:
    """
    The encoder is initialized with a list or set of real values. When called, the encoder will
    normalize its input based on the range of possible values given on initialization.

    Example:
    ---
        encode = LinearEncoder({3, 2, 1, 4, 12})
        encode([9]) # returns [0.75]
    """

    def __init__(self, data):
        self.max = max(data)

    def __call__(self, x):
        return [round(x[0]/self.max, 4)]

    def __len__(self):
        return 1


class MHEncoder:
    """
    Creates an encoder which converts a list of feature values into a multi-hot encoding.

    Example:
    ---
        encode = MHEncoder({'a', 'ab', 'jd', 'ke', 'ca', 'kk', 'p'})
        encode(['a', 'p', 'jd']) # returns array([1, 0, 1, 0, 0, 0, 1])
    """

    def __init__(self, unique):
        self.mapping = {k: i for i, k in enumerate(unique)}

    def __call__(self, data):
        return self.mhe([self.mapping[k] for k in data if k in self.mapping])

    def __len__(self):
        return len(self.mapping)

    def mhe(self, values):
        mhe = np.zeros(len(self))
        mhe[values] = 1
        return mhe


class Encoder():
    """The encoder wrapper class which allows to combine multiple encoders together into one."""

    def __init__(self, *encoders):
        """Receives a series of encoders which should all implement __call__ and __len__."""
        self.encoders = encoders

    def __call__(self, *params):
        """Encodes the list of features values into a single vector."""
        assert len(params) == len(self.encoders), '''Call must have the same number of parameters
            than number of encoders: {}, got {}'''.format(len(self.encoders), len(params))
        return np.concatenate([encoder(param) for encoder, param in zip(self.encoders, params)])

    def __len__(self):
        """Returns the number of bits used to encode the data."""
        return sum(map(len, self.encoders))
