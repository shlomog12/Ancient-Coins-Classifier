from bs4 import BeautifulSoup
import re
import requests

names = ['Alexander Jannaios', 'John Hyrkanos I', 'Mattathias Antigonos']

html_text = requests.get('https://www.coinarchives.com/a/results.php?search=Alexander+Jannaeus&s=0&upcoming=0&results=1000').text
soup = BeautifulSoup(html_text, 'lxml')

entries = soup.find_all('tr', {"id": re.compile("[0-9]")})
count = 0
for entry in entries:
    if entry.find(class_='rightborder').div.a and \
            entry.find(class_='leftborder').find(class_='R').find(class_='lottext').text.split('.')[1].split('(') \
                    [0].strip():
        print(f"[ {entry.get('id')} ]")
        print(entry.find(class_='leftborder').find(class_='R').find(class_='lottext').text.split('.')[1])
        print(entry.find(class_='rightborder').div.a.attrs['href'])
        count += 1
print(f'\n# of entries found: {count}\n')
# Mattathias Antigonus