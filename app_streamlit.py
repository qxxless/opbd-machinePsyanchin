import streamlit as st

import requests

st.set_page_config(

    page_title='Классификация фильмов',

    page_icon='🎬',

    layout='centered'

)

st.title('🎬 Определение жанра фильма')

st.write(

    'Введите описание фильма'

)

text = st.text_area(

    'Описание',

    height=180

)

if st.button('Определить жанр'):

    if text.strip() == '':

        st.warning(

            'Введите описание фильма'

        )

    else:

        response = requests.post(

            'http://127.0.0.1:8000/predict',

            json={

                'text': text

            }

        )

        result = response.json()

        st.success(

            f"Предсказанный жанр: {result['genre']}"

        )