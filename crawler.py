import requests
import json
import os
from multiprocessing.pool import ThreadPool
import io
from PIL import Image

BASE_DIR = "./images/Van_Gogh/"

if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR, exist_ok=True)

def getUrls(html: str):
    urls = []
    while True:
        start = html.find(startPatten)
        end = html.find(endPatten, start)
        if start < 0 or end < 0: break
        start += len(startPatten)
        end -= 1
        urls.append(html[start:end])
        html = html[end + 2:]
    return urls


def makeMinImageUrl(urls: list):
    for i in range(len(urls)):
        splitIndex = urls[i].find(" ")
        urls[i] = urls[i][:splitIndex]
    return urls

def downloadImg(url:str):
    imgResp = requests.get(url)
    imgData = imgResp.content
    img = Image.open(io.BytesIO(imgData)).convert("RGB")
    img.save(BASE_DIR + url[73:]+".jpg", "jpeg")
    print(f"save {url[73:]}")


pool = ThreadPool(32)

# BASE_URL = "https://www.vangoghmuseum.nl/zh/collection/search?q=&Artist=Vincent%20van%20Gogh&from={}"
BASE_URL = "https://www.vangoghmuseum.nl/zh/collection/search?q=&Artist=Vincent+van+Gogh&Genre=genre+picture%2Callegory%2Cartist%27s+portrait%2Cbathing+scene%2Chistory+%28visual+work%29%2Clandscape+%28representation%29%2Cmarine%2Creligion%2Criver+landscape%2Cstill+life%2Cstreet+scene%2C%E4%BA%BA%E5%83%8F%2C%E5%86%9C%E6%B0%91%E7%94%9F%E6%B4%BB%2C%E5%8A%A8%E7%89%A9%2C%E5%A4%B4%E5%83%8F%2C%E5%9F%8E%E5%B8%82%E6%99%AF%E8%A7%82%2C%E5%AE%A4%E5%86%85%E6%99%AF%2C%E5%BC%80%E8%8A%B1%2C%E6%9D%91%E9%95%87%E6%99%AF%E8%A7%82%2C%E6%B5%B7%E6%99%AF%2C%E8%82%96%E5%83%8F%E7%94%BB%2C%E8%87%AA%E7%84%B6%2C%E8%87%AA%E7%94%BB%E5%83%8F%2C%E8%8A%B1%E5%8D%89%2C%E8%A3%B8%E5%83%8F&from={}"

for i in range(0,766,24):
    respond = requests.get(BASE_URL.format(i))

    json.loads(respond.text)
    t = json.loads(respond.text)

    html = t['resultsHtml']
    startPatten = '<picture><source data-srcset="'
    endPatten = 'sizes=""'

    urls = getUrls(t['resultsHtml'])
    makeMinImageUrl(urls)

    for url in urls:
        pool.apply_async(downloadImg, args=(url,))
pool.close()
pool.join()
print("download done")