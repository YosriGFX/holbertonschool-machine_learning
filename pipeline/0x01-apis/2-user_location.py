#!/usr/bin/env python3
'''2. Rate me is you can!'''
import sys
import requests
from datetime import datetime


def get_location(url):
    '''returns the location of a specific user'''
    headers = {'Accept': 'application/vnd.github.v3+json'}
    request = requests.get(url)
    if request.status_code == 403:
        retry = request.headers['X-Ratelimit-Reset']
        seconds = int(retry) - int(datetime.now().timestamp())
        print('Reset in {} min'.format(seconds / 60))
    elif request.status_code == 404:
        print('Not found')
    elif request.status_code == 200:
        print(request.json()['location'])


if __name__ == '__main__':
    get_location(sys.argv[1])
