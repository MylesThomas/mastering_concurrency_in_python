# ch11/example6.py

import requests
import os

from timeit import default_timer as timer

def download_html(url):
    res = requests.get(url)
    # Ensure the directory exists
    os.makedirs('output', exist_ok=True)
    
    filename = 'output/%s.html' % os.path.basename(url)
    with open(filename, 'w', encoding="utf-8") as f:# Specify UTF-8 encoding here to fix error
        f.write(res.text)

urls = [
    'http://packtpub.com',
    'http://python.org',
    'http://docs.python.org/3/library/asyncio',
    'http://aiohttp.readthedocs.io',
    'http://google.com'
]

start = timer()

for url in urls:
    download_html(url)

print('Took %.2f seconds.' % (timer() - start))
