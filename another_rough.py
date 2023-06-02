import json

f= open("schemes.json")

data =json.load(f)


# print(data['schemes'][1])

for i in range(0,20):
    print(data['schemes'][i]['name'])