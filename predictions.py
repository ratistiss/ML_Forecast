import pandas as pd
from flask import Flask, jsonify, request
import pickle
import os

# load model
model = pickle.load(open('./predictions.pkl','rb'))

# app
app = Flask(__name__)


# routes
@app.route('/', methods=['POST'])
def predict():
    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)

    # predictions
    result = model.predict(data_df)

    # send back to browser
    output = {'results': int(result[0])}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(port = port, debug=True)