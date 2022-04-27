import asyncio
import aiohttp
import numpy as np
from PIL import Image
from io import BytesIO


async def fetch(session, url):
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                return (url, await resp.read())
                # catch http errors here
            else:
                print(resp.status)
                print(url)
                return (url, None)
    except aiohttp.ClientConnectionError as e:
        print(f'Connection Error: {e}')
        
async def fetch_concurrent(urls):
    img_arrays = list()
    loop = asyncio.get_running_loop()
    async with aiohttp.ClientSession() as session:
        tasks = []
        for u in urls:
            tasks.append(loop.create_task(fetch(session, u)))
            
        for result in asyncio.as_completed(tasks):
            url, img = await result
            if img is not None:
                img = np.array(Image.open(BytesIO(img)))
                img_arrays.append((url, img))
            else:
                img_arrays.append((url, img))
                
    return img_arrays