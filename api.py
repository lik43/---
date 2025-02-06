from fastapi import FastAPI,UploadFile, File, Response
from typing import List
import uvicorn
from sklearn.metrics import root_mean_squared_error, r2_score
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from sklearn.preprocessing import StandardScaler
import numpy as np

_MODELS = {}
_DATA_lst1 = {}
_DATA_lst = {}



@asynccontextmanager
async def lifespan(app: FastAPI):
    global _DATA_lst1
    _MODELS['linear'] = joblib.load('/home/c4/Рабочий стол/C4_M5/model.pkl')
    _DATA_lst1['df'] = pd.read_csv('/home/c4/Загрузки/BIG DATA/df_0.csv', nrows=1000)
    _DATA_lst1['target'] = pd.read_csv('/home/c4/Загрузки/BIG DATA/target_0.csv', nrows=1000)
    yield
    _MODELS.clear()

app = FastAPI(lifespan=lifespan)

@app.post('/load')
def load(files: List[UploadFile] = File(...)):
    global _DATA_lst
    data_lst = []
    for file in files:
        name = file.filename
        content = file.file.read()
        with open(name,'wb') as f:
            f.write(content)
        data = pd.read_csv(content, nrows=1000)
        data_lst.append(data)
    try:
        _DATA_lst['df'] = data_lst[0]
        _DATA_lst['target'] = data_lst[1]
    except:
        return 'Загрузить вначале датафрейм с признаками, затем с таргетами'

    return f'Файлы успешно звгружены'

@app.get('/preprocess')
def preprocess():
    sc = StandardScaler()

#Удалим столбцы, переданные col2drop
    _DATA_lst1['df'] = _DATA_lst1['df'].drop(['0','3','4','5','6','7','8','9','11',
                                '12','13','14','15','16','17','18','19',
                                '24','25','26','27','28','29'], axis=1)
# Произведем стандартизацию
    _DATA_lst1['df'] = pd.DataFrame(sc.fit_transform(_DATA_lst1['df']), columns=_DATA_lst1['df'].columns)
# Оптимизируем датафреймы, изменив тип данных
    _DATA_lst1['df'] = _DATA_lst1['df'].astype(np.float32)
    _DATA_lst1['target'] = _DATA_lst1['target'].astype(np.float32)

    return 'Обработка и сохрание прошло успешно!'

@app.get('/list_models')
def list_models():
    return list(_MODELS.keys())

@app.get('/choose_model')
def choose_model(model_type:str):
    if model_type == 'linear':
        return 'Выбрана линейная модель'
    else:
        return 'Извините, модель в стадии обучения'


@app.get('/predict')
def predict():
    pred = {}
    pred['y_pred'] = pd.DataFrame(_MODELS['linear'].predict(_DATA_lst1['df'])).head(5).to_dict(orient='records')
    return pred



if __name__ == '__main__':
    uvicorn.run('back:app',
                reload=True,
                port=8000,
                host='127.0.0.1'
                )