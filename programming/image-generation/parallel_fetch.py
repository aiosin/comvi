from concurrent.futures import ThreadPoolExecutor
import time

import requests

def fetch(a,const):
    url = 'http://httpbin.org/get?a={0}'.format(a)
    r = requests.get(url)
    result = r.json()['args']
    return (result,const)

start = time.time()

# if max_workers is None or not given, it will default to the number of processors, multiplied by 5
with ThreadPoolExecutor(max_workers=None) as executor:
    results = executor.submit(fetch, range(42),test='aylmao',timeout=None,chunksize=1)
    print(results)

print('time: {0}'.format(time.time() - start))