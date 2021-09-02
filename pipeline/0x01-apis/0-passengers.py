#!/usr/bin/env python3
'''0. Can I join?'''
import requests


def availableShips(passengerCount):
    '''returns the list of ships that can
    hold a given number of passengers'''
    ships = []
    request = {'next': 'https://swapi-api.hbtn.io/api/starships/'}
    while request['next'] is not None:
        url = request['next']
        request = requests.get(url).json()
        for row in request['results']:
            try:
                passenger = row['passengers']
                passenger = ''.join(passenger.split(','))
                passenger = int(passenger)
            except ValueError:
                passenger = 0
            if passenger >= passengerCount:
                ships.append(row['name'])
    return ships
