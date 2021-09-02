#!/usr/bin/env python3
'''0. Can I join?'''
import requests


def availableShips(passengerCount):
    '''returns the list of ships that can
    hold a given number of passengers'''
    ships = []
    url = 'https://swapi-api.hbtn.io/api/starships/'
    request = requests.get(url).json()
    result = request['results']
    while request['next'] is not None:
        url = request['next']
        request = requests.get(url).json()
        result += request['results']
    for row in result:
        try:
            if int(row['passengers']) >= passengerCount:
                ships.append(row['name'])
        except ValueError:
            pass
    return ships
