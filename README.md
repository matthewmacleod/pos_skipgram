# Readme

Welcome to the Part of Speech Skipgram Embeddings Repository

## PoS

The goal of this project is to create embeddings for part of speech tags generated from
conversational (chat) text. 

The part of speech embeddings are generated with spaCy.
This library normalizes a lot of punctuation, so we do not want to do any preprocessing
on the text. For spaCy annotation guide see:

https://spacy.io/api/annotation#pos-tagging

### Data

The training dataset is composed of various conversational sources.

- wikipedia (see gensim docs on how to download)
- twitter streamed 
    - stopwords
    - obsenities
    - rare words (occuring less than 10 times in toxic data
         https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion)
- youtube comments https://www.kaggle.com/datasnaek/youtube



### Run

First create pos data:
```
python src/convert_to_pos.py
```

Then to create embeddings, alter settings for part of speech embedding by tring something like this:
```
python src/skipgram_pos.py --model_name skipgram_pos --exp $jid --epochs 8 --embed_dim 20 --target_file pos_targets.txt
```


