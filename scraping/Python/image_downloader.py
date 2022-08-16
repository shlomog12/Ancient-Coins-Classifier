import time

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options


def download_image(url, name):
    # write image to file
    reponse = requests.get(url)
    if reponse.status_code == 200:
        with open(f"images/{name}", 'wb') as file:
            file.write(reponse.content)
            file.close()


file1 = open('AlexanderJannaeus.txt', 'r')
Lines = file1.readlines()

options = webdriver.ChromeOptions()
options.add_argument('user-data-dir=C:/Users/Adi Dahari/Desktop/FinalProject/scraping/Python/drivers')
driver = webdriver.Chrome(
    executable_path='C://Users/Adi Dahari/Desktop/FinalProject/scraping/Python/drivers/chromedriver.exe',
    chrome_options=options)
driver.get('https://www.acsearch.info/login.html')
time.sleep(10)
for i in Lines:
    print(i)
    driver.get(i)
    img = driver.find_element(by=By.TAG_NAME, value='img').get_attribute('src')
    name = i.split('=')[-1].split('\n')[0].strip() + '.jpg'
    download_image(img, name)
    time.sleep(5)
