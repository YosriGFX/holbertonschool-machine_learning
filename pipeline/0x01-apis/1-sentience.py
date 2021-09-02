#!/usr/bin/env python3
'''1. Where I am?'''
import requests


def sentientPlanets():
    '''returns the list of names of the
    home planets of all sentient species.'''
    sentient = []
    request = {'next': 'https://swapi-api.hbtn.io/api/species/'}
    while request['next'] is not None:
        request = requests.get(request['next']).json()
        for row in request['results']:
            url = row['homeworld']
            if url is not None:
                planet = requests.get(url).json()['name']
                if planet not in sentient:
                    sentient.append(planet)
    return sentient
