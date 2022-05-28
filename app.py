from flask import Flask, jsonify, render_template, request
import joblib
import os
import numpy as np

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")

@app.route('/predict',methods=['POST','GET'])
def result():

    item_weight= float(request.form['item_weight'])
    item_fat_content=float(request.form['item_fat_content'])
    item_visibility= float(request.form['item_visibility'])
    item_type= float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_establishment_year= float(request.form['outlet_establishment_year'])
    outlet_size= float(request.form['outlet_size'])
    outlet_location_type= float(request.form['outlet_location_type'])
    outlet_type= float(request.form['outlet_type'])

    X= np.array([[ item_weight,item_fat_content,item_visibility,item_type,item_mrp,
                  outlet_establishment_year,outlet_size,outlet_location_type,outlet_type ]])

    from pandas.core.arrays.sparse import SparseArray as sav
    path = os.path.join('C:' + os.sep, 'Users', 'bishe', 'Downloads', 'execute 2.0', 'final', 'Execute-2.0', 'models','sc.sav')
#sc_sav = pd.read_sav(path)
    scaler_path= path

    sc=joblib.load(scaler_path)

    X_std= sc.transform(X)
    from pandas.core.arrays.sparse import SparseArray as sav
    path = os.path.join('C:' + os.sep, 'Users', 'bishe', 'Downloads', 'execute 2.0', 'final', 'Execute-2.0', 'models','lr.sav')
    model_path=path

    model= joblib.load(model_path)

    Y_pred=model.predict(X_std)

    return jsonify({'Prediction': float(Y_pred)})

import webbrowser
from threading import Timer

if not os.environ.get("WERKZEUG_RUN_MAIN"):
    webbrowser.open_new('http://127.0.0.1:9457/')


app.run(debug='True',host="127.0.0.1", port=9457)

if name == 'main':
    main()
