import os, pickle

"""Savagely dumps/loads python objects inside file."""

def save(fp, *lists):
    with open(fp + '.tmp', 'wb') as f:
        pickle.dump(lists, f)
    os.replace(fp + '.tmp', fp)

def load(fp):
    if not os.path.isfile(fp):
        raise FileNotFoundError(fp)
    with open(fp, 'rb') as f:
        return pickle.load(f)
