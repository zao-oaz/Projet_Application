# ---------- Import Librairies ---------- #
import os
import flask
import pickle
import pandas as pd
from flask import Flask
from flask import request
from pathlib import Path


# ---------- Connexion Flask ---------- #

app = Flask(__name__)


# ---------- Import pickle/csv ---------- #

BASE_DIR = Path(__file__).resolve(strict=True).parent

# load model
def load_model():
    with open(f"{BASE_DIR}/model_xgb.pkl", "rb") as f:
        model = pickle.load(f)
        return model
        
# load csv
def load_csv():
    with open(f"{BASE_DIR}/data/clean_csv.csv", "rb") as f:
        csv = pd.read_csv(f, index_col=[0])
        csv.drop('TARGET', axis=1, inplace=True)
        return csv



# ---------- TESTS ---------- #

model = load_model()
csv = load_csv()

# ---------- HTML ---------- #

id_min = csv["SK_ID_CURR"].min()
id_max = csv["SK_ID_CURR"].max()

# ---------- Import images ---------- #

IMG_FOLDER = os.path.join('static', 'images')
app.config['IMGFOLDER'] = IMG_FOLDER

IMG_0 = os.path.join(app.config['IMGFOLDER'], 'mauvais_client.png')
IMG_1 = os.path.join(app.config['IMGFOLDER'], 'bon_client.png')

# ---------- Home page ---------- #

@app.route('/', methods=['GET'])
def home_page():
    if model is None:
        print("Model is None!!  There is no model.")

    return flask.render_template('home-page.html')

# ---------- Predict page ---------- #

@app.route('/home', methods=['POST', 'GET'])
def inputs_page():
    return flask.render_template('predict.html', id_min=id_min, id_max=id_max)

# ---------- Result page ---------- #

@app.route('/predict', methods=['POST'])
def prediction():
    if request.form['SK_ID_CURR']:
        id_client = request.form['SK_ID_CURR']
        id_client = int(id_client)
        test_score = csv[csv['SK_ID_CURR'] == id_client].values
        # test_score = csv.loc[id_client].values
        # test_score = test_score.reshape(1, -1)

        predictions = model.predict_proba(test_score)

        score = round((predictions[0][0]), 2)*100

        y_pred = model.predict(test_score)

        if y_pred == 1:
            prediction = 'est en difficultés de payement'
            prediction_img = IMG_0
        else:
            prediction = "n'est pas en difficultés de payement"
            prediction_img = IMG_1
            y_pred = 0   



    # ---------- results page ---------- #

    return flask.render_template('result.html',
                                 text=prediction,
                                 img=prediction_img,
                                 score=score/100,
                                 submission=id_client)


# ---------- Run app ---------- #

if __name__ == '__main__':
    app.run(debug=True, use_debugger=True)

