import sys, os
import time
import numpy as np
import tensorflow as tf

from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import zipfile

from collections import Counter
import random
import argparse

import mutils

import spacy
nlp = spacy.load('en')

def pos_spacy(text):
    doc = nlp(text)
    poss = [str(t.tag_) for t in doc]
    return ' '.join(poss)


def convert_text(clean=True, convert_text8=False):
    """ load texts
    """
    if convert_text8:
        with open('data/text8') as f:
            text = f.read()        
    
        if clean:
            text = mutils.clean_generic(text)
    
        pos = []
        words = text.split(" ")
        chunk_size = len(words) // 1000
        for i in range(0, len(words), chunk_size):
            if i+chunk_size <= len(words):
                t = words[i:i+chunk_size]
                p = pos_spacy(' '.join(t))
                pos.append(p)
    
    
        with open('data/text8_pos.txt', mode='w') as f:
            for line in pos:
                f.write(line+'\n')


    files = []
    with open('./data/targets_to_pos.txt') as f:
        for line in f:
            files.append(line.rstrip('\n'))

    for target in files:
        print('Processing file:', target)
        new_target = target.replace('.txt', '_pos.txt')
        lines = []
        with open('./data/' + target) as f:
            for line in f:
                cleaned = line.strip().rstrip("\n\r")
                if clean:
                    cleaned = mutils.clean_generic(cleaned)
                p = pos_spacy(cleaned)
                lines.append(p)

        with open('./data/' + new_target, mode='w') as f:
            for line in lines:
                f.write(line+'\n')

    return



def main():
    parser = argparse.ArgumentParser(description="convert text part of speech")
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--min_freq", type=int, default=10)
    parser.add_argument("--model_name", type=str, default='skipgram_conversational')
    parser.add_argument("--exp", help='experiment number', type=int, default=0)
    parser.add_argument('--seed', help='use 1 to generate random seed or int', type=int, default=1)

    args = parser.parse_args()
    print('Args:', args)

    # check to make sure necessary directories exist
    if not os.path.exists(args.data_dir):
        sys.exit('Error: create data directory')

    convert_text(False,False)

    return



if __name__ == "__main__":
    main()
