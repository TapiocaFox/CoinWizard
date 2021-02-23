#!/usr/bin/python3

import threading

class PairDataFetcher(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        print('PairDataFetcher thread started.')
