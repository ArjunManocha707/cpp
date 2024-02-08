from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
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

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Performance of the model: {mae:.2f}")
print(f"Accuracy of the model: {r2:.2f}")

def get_current_market_price(car_data):
    regression_model = LinearRegression()
    X_historical = pd.get_dummies(dataframe[features])
    y_historical = dataframe[target_variable]
    regression_model.fit(X_historical, y_historical)

    X_today = pd.get_dummies(car_data, columns=['Brand', 'fuel'])
    X_today = X_today.reindex(columns=X_historical.columns, fill_value=0)
    
    current_market_price = regression_model.predict(X_today)[0]
    return current_market_price

@app.route('/predict', methods=['POST'])
def predict() :
    if request.method == 'POST':
        print(request.form['year'])
        # car_today_data = pd.DataFrame({
        #     'Year': [input("Car's Brought Year: ")],
        #     'Brand': [input("Car Brand: ")],
        #     'Model': [input("Car's Model: ")],
        #     'driven': [input("Car driven in km: ")],
        #     'owners': [pd.to_numeric(input("Owner Number of car: "))],
        #     'fuel': [input("Fuel Type: ")],
        #     'car type': [input("Car Type: ")]
        # })
        # car_today_data = {
        #     'Year': (request.form['year'],),
        #     'Brand': (request.form['brand'],),
        #     'Model': (request.form['model'],),
        #     'driven': (request.form['driven'],),
        #     'owners': (pd.to_numeric(request.form['owners']),),
        #     'fuel': (request.form['fuel'],),
        #     'car type': (request.form['car_type'],)
        # }

        car_today_data = {
            'Year': [int(request.form['year'])],  # Convert to int if needed
            'Brand': [request.form['brand']],
            'Model': [request.form['model']],
            'driven': [float(request.form['driven'])],  # Convert to float if needed
            'owners': [pd.to_numeric(request.form['owners'])],
            'fuel': [request.form['fuel']],
            'car type': [request.form['car_type']]
        }

        # car_today_data_encoded = pd.get_dummies(car_today_data, columns=['Brand', 'fuel'])
        # car_today_data_encoded = car_today_data_encoded.reindex(columns=X_train.columns, fill_value=0)

        # single_prediction_rf = model.predict(car_today_data_encoded)
        # print(f"Predicted Car Price: {single_prediction_rf[0]:,.2f}")

        current_market_price = get_current_market_price(car_today_data)
        print(f"Current Market Price: {current_market_price}")

        if current_market_price < 0:
            print("Your car is very old and you are not supposed to drive it. You can have a fine from the police for driving this car. And this car is also not in a selling condition.")
        
        return render_template('result.html', prediction_rf=current_market_price)

if __name__ == '__main__':
    app.run(debug=True)