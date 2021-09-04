#!/usr/bin/env python3
'''30. List all documents in Python'''


def list_all(mongo_collection):
    '''A function that lists all documents in a collection'''
    return [doc for doc in mongo_collection.find()]
