# ---------- imports ---------- #

from app import load_model
from app import load_csv

# ---------- test 1 load model ---------- #

def test_load_model():
    model = load_model()
    assert model != None

# ---------- test 2 nb colonnes ---------- #

def test_nb_columns():
    csv = load_csv()
    assert len(csv.columns) == 21

# ---------- test 3 prediction ---------- #

# def test_predictions_range():
#     # Test if range of predictions stay between 1 and 100
#     csv = load_csv('clean_csv.csv')
#     model = load_model('model_lgbm.pkl')
#     predictions = model.predict(csv.drop('TARGET', axis=1))
#     assert ((predictions <= 1).all() & (predictions >= 0).all())