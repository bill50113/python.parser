# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 18:50:24 2023

@author: USER
"""

from bs4 import BeautifulSoup

import requests

header = {
    'User-Agent':
'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'
    }
    
    
url = 'https://www.esunbank.com/zh-tw/personal/deposit/rate/forex/foreign-exchange-rates'

data = requests.get(url,headers=header).text

soup = BeautifulSoup(data,'html.parser')

rates = soup.find(id = 'exchangeRate')

# print(rates)

tbody = rates.find('tbody')
trs = tbody.find_all('tr')[1:]  #直接進行切片,因為[0]是我們不要的

for row in trs:
    
    tds = row.find_all('td',recursive = False) #關閉遞迴(原本預設True) 
    if len(tds) == 4:                        #關閉的話就是find_all只會找到外面的不會再往內find
        print(tds[0].text.strip().split()[0])
        print('-'*40)
        print('即期匯率:',tds[1].text.strip())
        print('-'*40)
        print('網路匯率:',tds[2].text.strip())
        print('-'*40)
        print('現金匯率:',tds[3].text.strip())
        print('-'*40)
    
    # else:
    #     print(tds[0].text.strip())
    #     print(tds[1].text.strip())
    #     print(tds[2].text.strip())
    #     print('-'*40)