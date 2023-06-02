import requests

import geocoder
from geopy.geocoders  import Nominatim

g= geocoder.ip("me")
ans = g.latlng

latitude = ans[0]
longitude = ans[1]

API_KEY = "c0437e00966a480a847170734230106"

# api= "https://api.weatherapi.com/v1/current.json?key=c0437e00966a480a847170734230106&q=Bengaluru&aqi=no"

# another_api = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Bangalore?unitGroup=metric&key=XCD4Z28H76RWZ2VHQYQWGNC6T&contentType=json"

# resp = requests.get(another_api)

# print(resp.json())

api_x = "https://history.openweathermap.org/data/3.0/history/timemachine?lat={lat}&lon={lon}&dt={dt}&appid=2b88dc25610d6b24c02f6917ac58ef54"

api_y="https://history.openweathermap.org/data/2.5/aggregated/year?lat=35&lon=139&appid=2b88dc25610d6b24c02f6917ac58ef54"

resp = requests.get(api_y)

print(resp.status_code)


# response = requests.get(api)

# if response.status_code==200:
#     print("response recieved")
#     print(response.json())
# else:
#     print("unsuccesful response ")


# data = response.json()

# print("........location.........")

# print(data['location'])

# print("........currnet............")

# print(data['current']['temp_c'])
