from fastapi import Body, FastAPI
from pydantic import BaseModel
from typing import Optional, Dict
from itertools import islice
import time

from deepface import DeepFace
from get_images import fetch_concurrent
from celery_worker import predict_demographics


class Items(BaseModel):
    images: Dict
    cb_url: Optional[str]
    num_images_to_analyze: Optional[int] = 300

app = FastAPI()

@app.post("/predict")
async def predict_age_and_gender(items: Items):
    # get the image urls
    images = items.images
    max_images = items.num_images_to_analyze
    
    # Reduce total number of images to max_images
    if len(images) > max_images:
        images = dict(list(islice(images.items(), max_images)))
    img_urls = list(images.values())
    
    # async get the image from the url
    img_arrays = await fetch_concurrent(img_urls)

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
    
    return {'predictions': predictions}


@app.post("/predict_later")
async def predict_age_and_gender_later(items: Items):
    # get the image urls
    images = items.images
    cb_url = items.cb_url
    max_images = items.num_images_to_analyze
    
    # Reduce total number of images to max_images
    if len(images) > max_images:
        images = dict(list(islice(images.items(), max_images)))
    
    # Send image urls to celery worker
    print("Sending to worker")
    task = predict_demographics.apply_async((dict(images), cb_url), serializer = 'json')
    
    return({'Result' : 'Images being processed'})
