from fastapi import FastAPI

from pydantic import BaseModel

import pickle

with open(

    'model.pkl',

    'rb'

) as file:

    model = pickle.load(

        file

    )

with open(

    'vectorizer.pkl',

    'rb'

) as file:

    vectorizer = pickle.load(

        file

    )

app = FastAPI(

    title='Movie Genre Predictor'

)

class MovieText(

    BaseModel

):

    text: str


@app.get('/')

def home():

    return {

        'message':

        'API работает'

    }


@app.post('/predict')

def predict(

    movie: MovieText

):

    vector = vectorizer.transform(

        [movie.text]

    )

    prediction = model.predict(

        vector

    )

    return {

        'genre':

        prediction[0]

    }