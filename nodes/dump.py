"""Simple debugging nodes"""
import csv
import pandas as pd
from timeflux.core.node import Node

class Dump(Node):
    """Dump to CSV."""

    def __init__(self, fname='/tmp/dump.csv'):
        # self.writer = csv.writer(open(fname, 'a'))
        self.list_chunks = []
        self.fname = fname

    def update(self):
        if self.i.ready():
            self.list_chunks.append(self.i.data)
            # todo: save at the end only
            pd.concat(self.list_chunks, axis=0).to_csv(self.fname)
