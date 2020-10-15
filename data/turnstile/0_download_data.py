import urllib.request
from bs4 import BeautifulSoup
import os

dir = "raw"
if not os.path.isdir(dir):
    os.mkdir(dir)

with urllib.request.urlopen(
    "http://web.mta.info/developers/turnstile.html"
) as response:
    html = response.read()
    soup = BeautifulSoup(html)
    for link in soup.select("#contentbox > .container > .last a"):
        url = "http://web.mta.info/developers/" + link.get("href")
        if "turnstile_141011.txt" in url:
            # Only take data newer than this
            break
        os.system("wget {} -P {}".format(url, dir))
