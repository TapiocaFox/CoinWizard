#!/usr/bin/python3

import threading
from PyInquirer import prompt, style_from_dict
import asyncio
import time
# custom_style_1 = style_from_dict({
#     "separator": '#cc5454',
#     "questionmark": '#673ab7 bold',
#     "selected": '#cc5454',  # default
#     "pointer": '#673ab7 bold',
#     "instruction": '',  # default
#     "answer": '#f44336 bold',
#     "question": '',
# })

class Cli(threading.Thread):
    def __init__(self, on):
        threading.Thread.__init__(self)
        on('prompt', self.prompt)

    def prompt(self, questions):
        print('Cli fafdasf started.')
        answer = prompt(questions)
        return answer

    def run(self):
        print('Cli thread started.')
