from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

dataframe = pd.read_csv("car_data.csv")

features = ['Year', 'Brand', 'Model', 'driven ', 'owners', 'fuel', 'car type']
target_variable = 'prices'

X = pd.get_dummies(dataframe[features])
y = dataframe[target_variable]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

def get_current_market_price(car_data):
    regression_model = LinearRegression()
    X_historical = X
    y_historical = y
    regression_model.fit(X_historical, y_historical)

    car_today_data_encoded = pd.get_dummies(pd.DataFrame(car_data, index=[0]), columns=['Brand', 'fuel'])
    car_today_data_encoded = car_today_data_encoded.reindex(columns=X_historical.columns, fill_value=0)
    
    current_market_price = regression_model.predict(car_today_data_encoded)[0]
    return current_market_price

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        car_today_data = {
            'Year': int(request.form['year']),
            'Brand': request.form['brand'],
            'Model': request.form['model'],
            'driven': float(request.form['driven']),
            'owners': int(request.form['owners']),
            'fuel': request.form['fuel'],
            'car type': request.form['car_type']
        }

        current_market_price = get_current_market_price(car_today_data)

        if current_market_price < 0:
            print("Your car is very old and you are not supposed to drive it. You can have a fine from the police for driving this car. And this car is also not in a selling condition.")
        
        return render_template('result.html', prediction_rf=current_market_price, year=request.form['year'], brand=request.form['brand'], model=request.form['model'], driven=request.form['driven'], owners=request.form['owners'], fuel=request.form['fuel'], car_type=request.form['car_type'])

if __name__ == '__main__':
    app.run(debug=True)