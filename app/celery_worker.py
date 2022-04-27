import os
import time
import requests
import json
import functools
import asyncio
from deepface import DeepFace

from celery import Celery
from dotenv import load_dotenv
from get_images import fetch_concurrent

load_dotenv(".env")

celery = Celery(__name__)
celery.conf.broker_url = os.environ.get("CELERY_BROKER_URL")
celery.conf.result_backend = os.environ.get("CELERY_RESULT_BACKEND")

def sync(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.get_event_loop().run_until_complete(f(*args, **kwargs))
    return wrapper

# task to get images from url
@celery.task(name = "predict_demographics")
@sync
async def predict_demographics(images, cb_url):
    img_urls = list(images.values())
    start_img_ret = time.time()
    img_arrays = await fetch_concurrent(img_urls)
    end_img_ret = time.time()
    time_to_get_images = end_img_ret - start_img_ret

    start_predict = time.time()
    print(f'Image retrieval took {time_to_get_images} seconds.')
    
    total_images =  len(img_arrays)
    actions = ['age', 'gender']
    # options for face detection.  mtcnn is default and relatively fast.  retina is slower but more accurate
    # alternative option is to use a seperate face detector and then pass individual face object to the analyze with detector_backend = 'skip'
    backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
    
    predictions = dict()
    for i, (url, img_array) in enumerate(img_arrays):
        if img_array is None:
            # TODO return that this url had 400 status code
            predictions[url] = {'age': None,
                                'gender' : None}
        else:
            try:
                obj = DeepFace.analyze(img_path = img_array,
                                    actions = actions,
                                    detector_backend = backends[3],
                                    prog_bar = False)
                
                predictions[url] = {'age': obj['age'],
                                    'gender' : obj['gender']}
            except ValueError:
                predictions[url] = {'age': None,
                                    'gender' : None}
        if i !=0 and i % 50 == 0:
            print(f'{i+1} images out of {total_images} analyzed.\n  A face was detected in {round(len(predictions)/ total_images *100, 2)}% of the images.')
    
    # extract just the items with a prediction to calculate what % of images had a face identified
    total_predictions = {k:v for k, v in  predictions.items() if v['age'] != None}
    print(f'{i+1} images out of {total_images} analyzed.\n  A face was detected in {round(len(total_predictions)/ total_images *100, 2)}% of the images.')

    # Map predictions back to original channel_id to return
    predictions = {k: predictions[v] for k,v in images.items()}
    
    end_predict = time.time()
    time_to_predict = end_predict - start_predict
    print(predictions)
    # Return predictions to cb_url
    payload = json.dumps({
        "msgtype": "text",
        "text": {
            "content": f"Prediction: {predictions}.  It took {time_to_get_images} seconds to get the images and {time_to_predict} to make the predictions"
            }
        })
    headers = {
        'Content-Type': 'application/json'
        }
    response = requests.request("POST", cb_url, headers=headers, data=payload)
    print(response.text)
    return