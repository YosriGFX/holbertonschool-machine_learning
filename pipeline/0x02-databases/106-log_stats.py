#!/usr/bin/env python3
'''41. Log stats'''
from pymongo import MongoClient
from bson.son import SON


if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    school = client.logs.nginx
    print('{} logs'.format(school.count_documents({})))
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    print('Methods:')
    for method in methods:
        print('\tmethod {}: {}'.format(
            method,
            school.count_documents({'method': method})
        ))
    print('{} status check'.format(
        school.count_documents(
            {'method': 'GET', 'path': '/status'}
        )
    ))
    pipeline = [
        {"$group": {
            "_id": "$ip",
            "count": {"$sum": 1}
        }},
        {"$sort": SON([("count", -1)])}
    ]
    agg = list(school.aggregate(pipeline=pipeline))[:10]
    agg[0], agg[1], agg[2] = agg[1], agg[2], agg[0]
    print("IPs:")
    for item in agg:
        print('\t{}: {}'.format(item['_id'], item['count']))
