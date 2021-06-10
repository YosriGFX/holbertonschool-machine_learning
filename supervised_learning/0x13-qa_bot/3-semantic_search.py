#!/usr/bin/env python3
'''3. Semantic Search'''
import tensorflow_hub as hub
import numpy as np
import os


def semantic_search(corpus_path, sentence):
    '''performs semantic search on a corpus of documents'''
    embed = hub.load(
        'https://tfhub.dev/google/universal-sentence-encoder-large/5'
    )
    Search = [sentence]
    documents = []
    for file in os.listdir(corpus_path):
        if file.endswith(".md"):
            filename = os.path.join(corpus_path, file)
            with open(filename) as f:
                doc = f.read()
                documents.append(doc)
    return documents[
        np.argmax(
            np.dot(
                embed(Search).numpy(),
                (embed(documents).numpy()).T
            )
        )
    ]
