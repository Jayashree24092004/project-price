import os
import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    data = pd.read_csv(filepath)
    X = data[['Month']]
    y = data['Price']

    model = LinearRegression()
    model.fit(X, y)

    # Predict for next 3 months
    future_months = pd.DataFrame({'Month': [X['Month'].max() + i for i in range(1, 4)]})
    future_prices = model.predict(future_months)

    predictions = list(zip(future_months['Month'], future_prices.round(2)))

    return render_template('result.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
