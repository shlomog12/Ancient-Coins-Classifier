import shutil
import time
from os import path
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
# session = HTMLSession()
#
#
# resp = session.get(
#     "https://www.acsearch.info/search.html?term=+John+Hyrcanus+I&category=1-2&lot=&thesaurus=1&images=1&en=1&de=1&fr=1&it=1&es=1&ot=1&currency=usd&order=0")
#
# # Run JavaScript code on webpage
# html = resp.html.render()
# html = resp.html.html

browser = webdriver.Chrome('C://Users/Adi Dahari/Desktop/FinalProject/scraping/Python/drivers/chromedriver.exe')
browser.get("https://www.acsearch.info/search.html?term=John+Hyrcanus+I&category=1-2&lot=&thesaurus=1&images=1&en=1&de=1&fr=1&it=1&es=1&ot=1&currency=usd&order=0")
time.sleep(1)

body = browser.find_element_by_tag_name('html')
pgdown = 120

while pgdown:
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.2)
    pgdown -= 1
print(body)

html = browser.page_source

soup = BeautifulSoup(html, 'lxml')
entries = soup.find_all('div', class_='auction-lot')

count = 0
imgs = []
for entry in entries:
    entry_id = entry.get('data-id')
    name = entry.find('div', class_='lot-desc').text.strip().lower()
    img = entry.find('img').get('src')
    img = img.replace('.s.', '.m.')
    if 'prutah' in name and 'half prutah' not in name:
        path = 'https://www.acsearch.info/' + img
        r = requests.get(path, stream=True)
        print(f'downloading {path}')
        # Check if the image was retrieved successfully
        if r.status_code == 200:
            # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
            r.raw.decode_content = True

            # Open a local file with wb ( write binary ) permission.
            with open(f'images/John Hyrcanus I/{entry_id}.jpg', 'wb') as f:
                shutil.copyfileobj(r.raw, f)
            print(f"{f'images/John Hyrcanus I/{entry_id}.jpg'} written")

