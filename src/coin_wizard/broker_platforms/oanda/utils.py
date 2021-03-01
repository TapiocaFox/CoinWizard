#!/usr/bin/python3

def translate_pair_to_oanda(pair_name):
    return pair_name[0:3].upper() + '_' + pair_name[4:7].upper()

def translate_pair_to_oanda(pair_name):
    if(pair_name[3] == '_'):
        return pair_name[0:3].upper() + '_' + pair_name[4:7].upper()
    return pair_name
