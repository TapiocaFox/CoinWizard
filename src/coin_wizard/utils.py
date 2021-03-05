#!/usr/bin/python3

def translate_pair_to_splited(pair_name):
    if len(pair_name) != 6:
        return pair_name
    return pair_name[0:3].upper() + '_' + pair_name[3:6].upper()

def translate_pair_to_unsplited(pair_name):
    if len(pair_name) != 7:
        return pair_name
    return pair_name[0:3].lower()+ pair_name[4:7].lower()
