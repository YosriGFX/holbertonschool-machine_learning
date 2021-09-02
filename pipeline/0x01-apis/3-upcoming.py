#!/usr/bin/env python3
'''3. What will be next?'''
import requests


def get_launches():
    '''return launches'''
    time_now = datetime.now().timestamp()
    url = 'https://api.spacexdata.com/v4/launches/upcoming'
    request = requests.get(url).json()
    latest = {}
    for row in request:
        if 'date_unix' not in latest or row['date_unix'] < latest['date_unix']:
            latest = row
    launch_name = latest['name']
    launch_date = latest['date_local']
    rocket_name = requests.get(
        'https://api.spacexdata.com/v4/rockets/' + latest['rocket']
    ).json()['name']
    launchpad = requests.get(
        'https://api.spacexdata.com/v4/launchpads/' + latest['launchpad']
    ).json()
    launchpad_name = launchpad['name']
    launchpad_locality = launchpad['locality']
    print('{} ({}) {} - {} ({})'.format(
        launch_name,
        launch_date,
        rocket_name,
        launchpad_name,
        launchpad_locality
    ))


if __name__ == '__main__':
    get_launches()
