#!/usr/bin/env python3
'''0. Can I join?'''
import requests


def availableShips(passengerCount):
    '''returns the list of ships that can
    hold a given number of passengers'''
    ships = []
    result = []
    request = {'next': 'https://swapi-api.hbtn.io/api/starships/'}
    while request['next'] is not None:
        url = request['next']
        request = requests.get(url).json()
        result += request['results']
    for row in result:
        try:
            passenger = int(row['passengers'])
        except ValueError:
            passenger = 0
        if passenger >= passengerCount:
            ships.append(row['name'])
    return ships
