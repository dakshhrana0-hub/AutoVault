import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
import time

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
}

# Expanded brand list
brands = ['maruti suzuki', 'hyundai', 'honda', 'toyota', 'tata', 'mahindra', 'mercedes benz','ford','volkswagen','audi','nissan' ,'bmw', 'kia']
BASE_URL = "https://www.olx.in/delhi_g4058659/cars_c84?filter=make_eq_{}&page={}"
car_data = []

def extract_car_info(card):
    title = card.find('div',{'class':'_2Gr10'}).text.strip()
    
    link = "https://www.olx.in" + card.find('a')['href']
    
    image = card.find('img')['src']
    location_item = card.find('div', {'class': '_3VRSm'})
    location =location_item.find('span').text.strip()
    
    price = card.find('span', {"class":"_1zgtX"}).text.strip()
    info = card.find('div',{'class':'_21gnE'}).text.strip()

    
    detail_response = requests.get(link, headers=HEADERS)
    detail_soup = BeautifulSoup(detail_response.text, 'html.parser')

    # Extract only the data you mentioned: div._3rMkw
    data_blocks = detail_soup.find_all('h2', {'class': '_3rMkw'})
    detail_data = None
    if data_blocks:
        detail_data = " , ".join(block.text.strip() for block in data_blocks)
    
    print(detail_data)
    
    return {
        'Title': title,
        'Link': link,
        'Location': location,
        'Price': price,
        'Information': info,
        'Image':image,
        'Detail':detail_data
    }

for brand in brands:
    for i in range(2,25):
        print(f"Scraping: {brand}")
        url = BASE_URL.format(brand,i)
        response = requests.get(url, headers=HEADERS)
        print(response.status_code)
        soup = BeautifulSoup(response.text, 'html.parser')
        listings = soup.find_all('li', {'class': '_3V_Ww'})
        
        for card in listings:
            info = extract_car_info(card)
            info['Brand'] = brand
            if info:
                car_data.append(info)

    time.sleep(2)

df = pd.DataFrame(car_data)
print(df.head())

path = os.path.join(os.getcwd(),'data','raw_data','olx_listings_delhi.csv')
df.to_csv(path, index=False)