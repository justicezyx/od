import requests
import json

# Define the URL
url = "https://api.stackexchange.com/2.2/users"

# Send the POST request with parameters to fetch the top 10 users
params = {
    'site': 'stackoverflow',
    'page': 1,
    'pagesize': 1,
    'order': 'desc',
    'sort': 'reputation',
    'site': 'stackoverflow',
}
response = requests.get(url, params=params)

# Display the obtained JSON payload
if response.status_code != 200:
    print(f"Request failed with status code: {response.status_code}")
    exit(1)

json_data = response.json()

# Extract display_name and profile_image from items
for item in json_data.get('items', []):
    display_name = item.get('display_name')
    profile_image = item.get('profile_image')
    print(f"Display Name: {display_name}")
    print(f"Profile Image: {profile_image}")
