import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
}

# Base URL with price filter, page placeholder
BASE_URL = "https://www.olx.in/delhi_g4058659/cars_c84?filter=price_min_8000000&page={}"
car_data = []

def extract_car_info(card):
    # Basic info from listing card
    title = card.find('div', {'class': '_2Gr10'}).text.strip()
    link = "https://www.olx.in" + card.find('a')['href']
    image = card.find('img')['src']
    location_item = card.find('div', {'class': '_3VRSm'})
    location = location_item.find('span').text.strip()
    price = card.find('span', {"class": "_1zgtX"}).text.strip()
    info = card.find('div', {'class': '_21gnE'}).text.strip()

    # Request detail page
    detail_response = requests.get(link, headers=HEADERS)
    detail_soup = BeautifulSoup(detail_response.text, 'html.parser')

    # Extract detail data from h2._3rMkw
    data_blocks = detail_soup.find_all('h2', {'class': '_3rMkw'})
    detail_data = None
    if data_blocks:
        detail_data = " , ".join(block.text.strip() for block in data_blocks)

    # Extract owner info from div._3VRXh
    owner_block = detail_soup.find('div', {'class': '_3VRXh'})
    owner = owner_block.text.strip() if owner_block else None
    print(title)
    return {
        'Title': title,
        'Link': link,
        'Location': location,
        'Price': price,
        'Information': info,
        'Image': image,
        'Detail': detail_data,
        'Owner': owner
    }

# Loop over pages only
for i in range(2, 30):
    print(f"Scraping page {i}")
    url = BASE_URL.format(i)
    response = requests.get(url, headers=HEADERS)
    print(response.status_code)
    soup = BeautifulSoup(response.text, 'html.parser')
    listings = soup.find_all('li', {'class': '_3V_Ww'})

    for card in listings:
        info = extract_car_info(card)
        if info:
            car_data.append(info)

    time.sleep(2)

# Save to CSV
df = pd.DataFrame(car_data)
print(df.head())

path = os.path.join(os.getcwd(), 'data', 'raw_data', 'olx_luxe_listings_delhi.csv')
os.makedirs(os.path.dirname(path), exist_ok=True)
df.to_csv(path, index=False)