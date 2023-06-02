
import geocoder

from geopy.geocoders import Nominatim
g= geocoder.ip("me")

print(g.latlng)

ans = g.latlng

latitude = ans[0]
longitude = ans[1]

print(latitude, longitude, ans)

geolocator = Nominatim(user_agent="geoapiExercises")

location = geolocator.reverse(str(latitude)+","+str(longitude))

address = location.raw['address']

print(location)

print(address)

print(address['city'])
