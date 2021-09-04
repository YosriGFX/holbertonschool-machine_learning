#!/usr/bin/env python3
'''4. How many by rocket?'''
import requests


if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4/rockets'
    rockets = [
        rocket['name'] for rocket in requests.get(url).json()
    ]
    url = 'https://api.spacexdata.com/v3/launches'
    for rocket in rockets:
        params = {'rocket_name': rocket}
        print('{}: {}'.format(
            rocket,
            len(requests.get(url, params=params).json())
        ))
