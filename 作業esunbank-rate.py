# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:56:18 2023

@author: s1211
"""

from bs4 import BeautifulSoup

import requests

header = {
    'User-Agent':
'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'
    }
    
    
url = 'https://www.esunbank.com/zh-tw/personal/deposit/rate/forex/foreign-exchange-rates'

data = requests.get(url,headers=header).text
# print(data)

soup = BeautifulSoup(data,'html.parser')

rate = soup.find('table')
# print(rate)

tbody = rate.find('tbody')
trs = tbody.find_all('tr')
thead = rate.find('tr')
ths = thead.find_all('th')
b = rate.text.strip()
b = b.split()
print()
print()
print('玉山銀行即期匯率')
print()
print('▼'*30)

for row in trs[1::2]:
    # print(row)
    tds = row.find_all('td')

    currency = tds[0].text.strip()    #  .strip() 去前後空白
    currency = currency.split()      #   預設以空白去切割

    

    q = row.find_all('div')
    q1 = q.
    d1 = row.find_all('div')
    div1 = d1[5].text.strip()
    div1 = div1.split()
    d2 = row.find_all('div')
    div2 = d2[6].text.strip()
    div2 = div2.split()
    d3 = row.find_all('div')
    div3 = d3[7].text.strip()
    div3 = div3.split()
    d4 = row.find_all('div')
    div4 = d4[8].text.strip()
    div4 = div4.split()
    d5 = row.find_all('div')
    div5 = d5[9].text.strip()
    div5 = div5.split()
    d6 = row.find_all('div')
    div6 = d6[10].text.strip()
    div6 = div6.split()
    div5.append('None')
    div6.append('None')
    # print(div1)
    # print('-'*60)
    # print(currency)
    # print('*'*30)
    # print(currency[0])
    # print(currency[1])
    # print(currency[2])
    # print(currency[3])
    # print(tds)
    # print(tds[0].text.strip())       #  .strip() 去前後空白
    
    
    print(' '*50,currency[0],currency[1])
    print()
    # print(currency[1])
    # print(tds[1].text.strip())
    print(ths[1].text)
    print(tds[2].text.strip(),':',div1[0])
    print(tds[3].text.strip(),':',div2[0])
    # print(tds[4].text.strip())
    print()
    print(ths[2].text)
    print(tds[2].text.strip(),':',div3[0])
    print(tds[3].text.strip(),':',div4[0])
    # print(tds[5].text.strip())
    print()
    print(ths[3].text)
    print(tds[2].text.strip(),':',div5[0])
    print(tds[3].text.strip(),':',div6[0])
    # print(tds[6].text.strip())

    print('▲'*30)
    print()
    