from django.shortcuts import render
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import pytz
import os

API_KEY = "aeae13a78cb0115957d35610b9497453"
BASE_URL = "https://api.openweathermap.org/data/2.5/"


# 1
def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    return {
        "city": data["name"],
        "current_temp": round(data["main"]["temp"]),
        "feels_like": round(data["main"]["feels_like"]),
        "temp_min": round(data["main"]["temp_min"]),
        "temp_max": round(data["main"]["temp_max"]),
        "humidity": round(data["main"]["humidity"]),
        "description": data["weather"][0]["description"],
        "country": data["sys"]["country"],
        "Wind_Gust_Dir": data["wind"]["deg"],
        "pressure": data["main"]["pressure"],
        "Wind_Gust_Speed": data["wind"]["speed"],
        "clouds":data['clouds']['all'],
        "Visibility":data['visibility']
    }


# 2
def read_historical_data(filename):
    df = pd.read_csv(filename)
    df = df.dropna()
    df = df.drop_duplicates()
    return df


# 3
def prepare_data(data):
    le = LabelEncoder()
    data["WindGustDir"] = le.fit_transform(data["WindGustDir"])
    data["RainTomorrow"] = le.fit_transform(data["RainTomorrow"])

    X = data[
        [
            "MinTemp",
            "MaxTemp",
            "WindGustDir",
            "WindGustSpeed",
            "Humidity",
            "Pressure",
            "Temp",
        ]
    ]
    y = data["RainTomorrow"]
    return X, y, le


# 4
def train_rain_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("mean_squared_error For Rain Model")
    print(mean_squared_error(y_test, y_pred))
    return model


# 5
def prepare_regression_data(data, feature):
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i + 1])
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    return X, y


# 6
def train_regression_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


# 7
def predict_future(model, current_value):
    predictions = [current_value]
    for i in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))
        predictions.append(next_value[0])
    return predictions[1:]


# Weather anylysis function


def weather_view(request):
    if request.method == "POST":
        city = request.POST.get("city")
        current_weather = get_current_weather(city)

        # load historical data


        csv_path = os.path.join("C:\\lucifer\\rainfall_prediction-2\\Weather_prediction\\weather.csv")
        historical_data = read_historical_data(csv_path)

        X, y, le = prepare_data(historical_data)

        rain_model = train_rain_model(X, y)

        wind_deg = current_weather["Wind_Gust_Dir"] % 360
        compass_points = [
            ("N", 0, 11.25),
            ("NNE", 11.25, 33.75),
            ("NE", 33.75, 56.25),
            ("ENE", 56.25, 78.75),
            ("E", 78.75, 101.25),
            ("ESE", 101.25, 123.75),
            ("SE", 123.75, 146.25),
            ("SSE", 146.25, 168.75),
            ("S", 168.75, 191.25),
            ("SSW", 191.25, 213.75),
            ("SW", 213.75, 236.25),
            ("WSW", 236.25, 258.75),
            ("W", 258.75, 281.25),
            ("WNW", 281.25, 303.75),
            ("NW", 303.75, 326.25),
            ("NNW", 326.25, 348.75),
        ]
        compass_direction = next(
            point for point, start, end in compass_points if start <= wind_deg < end
        )
        compass_direction_encoded = (
            le.transform([compass_direction])[0]
            if compass_direction in le.classes_
            else -1
        )

        current_data = {
            "MinTemp": current_weather["temp_min"],
            "MaxTemp": current_weather["temp_max"],
            "WindGustDir": compass_direction_encoded,
            "WindGustSpeed": current_weather["Wind_Gust_Speed"],
            "Humidity": current_weather["humidity"],
            "Pressure": current_weather["pressure"],
            "Temp": current_weather["current_temp"],
        }

        current_df = pd.DataFrame([current_data])
        rain_prediction = rain_model.predict(current_df)[0]

        X_temp, y_temp = prepare_regression_data(historical_data, "Temp")
        X_hum, y_hum = prepare_regression_data(historical_data, "Humidity")
        temp_model = train_regression_model(X_temp, y_temp)
        hum_model = train_regression_model(X_hum, y_hum)

        future_temp = predict_future(temp_model, current_weather["temp_min"])
        future_humidity = predict_future(hum_model, current_weather["humidity"])

        import pytz

        timezone = pytz.timezone("Asia/Karachi")
        now = datetime.now(timezone)
        next_hour = now + timedelta(hours=1)
        next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
        future_times = [
            (next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)
        ]

        # store each values sepratelly
        time1, time2, time3, time4, time5 = future_times
        temp1, temp2, temp3, temp4, temp5 = future_temp
        hum1, hum2, hum3, hum4, hum5 = future_humidity

        # pass the data to the template

        context = {
            "location": city,
            "current_temp": current_weather["current_temp"],
            "min_temp": current_weather["temp_min"],
            "max_temp": current_weather["temp_max"],
            "feels_like": current_weather["feels_like"],
            "humidity": current_weather["humidity"],
            "clouds": current_weather["clouds"],
            "visibility": current_weather["Visibility"],
            "description": current_weather["description"],
            'city':current_weather["city"],
            "country": current_weather["country"],
            
            'time':datetime.now(),
            "date": datetime.now().strftime("%B ,%D %y"),

            'wind':current_weather['Wind_Gust_Speed'],
            "pressure": current_weather["pressure"],
            # Forecast for next 5 hours
            "time1": time1,
            "time2": time2,
            "time3": time3,
            "time4": time4,
            "time5": time5,
            
            "temp1": f"{round(temp1,1)}",
            "temp2": f"{round(temp2,2)}",
            "temp3": f"{round(temp3,3)}",
            "temp4": f"{round(temp4,4)}",
            "temp5": f"{round(temp5,5)}",
            
            "hum1": f"{round(hum1,1)}",
            "hum2": f"{round(hum2,2)}",
            "hum3": f"{round(hum3,3)}",
            "hum4": f"{round(hum4,4)}",
            "hum5": f"{round(hum5,5)}",
        }
        
        return render(request, 'weather.html', context)
    return render(request, 'weather.html')

# from django.shortcuts import render
# import requests
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# from datetime import datetime, timedelta
# import pytz
# import os

# API_KEY = "aeae13a78cb0115957d35610b9497453"
# BASE_URL = "https://api.openweathermap.org/data/2.5/"

# def get_current_weather(city):
#     url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
#     response = requests.get(url)
#     data = response.json()
#     return {
#         "city": data["name"],
#         "current_temp": round(data["main"]["temp"]),
#         "feels_like": round(data["main"]["feels_like"]),
#         "temp_min": round(data["main"]["temp_min"]),
#         "temp_max": round(data["main"]["temp_max"]),
#         "humidity": round(data["main"]["humidity"]),
#         "description": data["weather"][0]["description"],
#         "country": data["sys"]["country"],
#         "Wind_Gust_Dir": data["wind"]["deg"],
#         "pressure": data["main"]["pressure"],
#         "Wind_Gust_Speed": data["wind"]["speed"],
#         "clouds": data['clouds']['all'],
#         "Visibility": data['visibility']
#     }

# def read_historical_data(filename):
#     df = pd.read_csv(filename)
#     df = df.dropna()
#     df = df.drop_duplicates()
#     return df

# def prepare_data(data):
#     le = LabelEncoder()
#     data["WindGustDir"] = le.fit_transform(data["WindGustDir"])
#     data["RainTomorrow"] = le.fit_transform(data["RainTomorrow"])
#     X = data[["MinTemp", "MaxTemp", "WindGustDir", "WindGustSpeed", "Humidity", "Pressure", "Temp"]]
#     y = data["RainTomorrow"]
#     return X, y, le

# def train_rain_model(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     print("mean_squared_error For Rain Model")
#     # print(mean_squared_error(y_test, model.predict(X_test))) # Removed prediction here
#     return model

# def prepare_regression_data(data, feature):
#     X, y = [], []
#     for i in range(len(data) - 1):
#         X.append(data[feature].iloc[i])
#         y.append(data[feature].iloc[i + 1])
#     X = np.array(X).reshape(-1, 1)
#     y = np.array(y)
#     return X, y

# def train_regression_model(X, y):
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X, y)
#     return model

# def predict_future(model, current_value):
#     predictions = [current_value]
#     for _ in range(5):
#         next_value = model.predict(np.array([[predictions[-1]]]))
#         predictions.append(next_value[0])
#     return predictions[1:]

# def weather_view(request):
#     context = {} # Initialize an empty context
#     if request.method == "POST":
#         city = request.POST.get("city")
#         if city: # Only proceed if a city is entered
#             try:
#                 current_weather = get_current_weather(city)

#                 csv_path = os.path.join("C:\\lucifer\\rainfall_prediction-2\\ weather.csv")
#                 historical_data = read_historical_data(csv_path)
#                 X, y, le = prepare_data(historical_data)
#                 rain_model = train_rain_model(X, y)

#                 wind_deg = current_weather["Wind_Gust_Dir"] % 360
#                 compass_points = [...] # Your compass points list
#                 compass_direction = next((point for point, start, end in compass_points if start <= wind_deg < end), "")
#                 compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1

#                 current_data = {
#                     "MinTemp": current_weather["temp_min"],
#                     "MaxTemp": current_weather["temp_max"],
#                     "WindGustDir": compass_direction_encoded,
#                     "WindGustSpeed": current_weather["Wind_Gust_Speed"],
#                     "Humidity": current_weather["humidity"],
#                     "Pressure": current_weather["pressure"],
#                     "Temp": current_weather["current_temp"],
#                 }
#                 current_df = pd.DataFrame([current_data])
#                 rain_prediction = rain_model.predict(current_df)[0]

#                 X_temp, y_temp = prepare_regression_data(historical_data, "Temp")
#                 X_hum, y_hum = prepare_regression_data(historical_data, "Humidity")
#                 temp_model = train_regression_model(X_temp, y_temp)
#                 hum_model = train_regression_model(X_hum, y_hum)

#                 future_temp = predict_future(temp_model, current_weather["temp_min"])
#                 future_humidity = predict_future(hum_model, current_weather["humidity"])

#                 timezone = pytz.timezone("Asia/Karachi")
#                 now = datetime.now(timezone)
#                 next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
#                 future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

#                 time1, time2, time3, time4, time5 = future_times
#                 temp1, temp2, temp3, temp4, temp5 = future_temp
#                 hum1, hum2, hum3, hum4, hum5 = future_humidity

#                 context = {
#                     "location": city,
#                     "current_temp": current_weather["current_temp"],
#                     "min_temp": current_weather["temp_min"],
#                     "max_temp": current_weather["temp_max"],
#                     "feels_like": current_weather["feels_like"],
#                     "humidity": current_weather["humidity"],
#                     "clouds": current_weather["clouds"],
#                     "visibility": current_weather["Visibility"],
#                     "description": current_weather["description"],
#                     'city': current_weather["city"], # Corrected access
#                     "country": current_weather["country"],
#                     'time': now,
#                     "date": now.strftime("%B ,%d %Y"), # Corrected format
#                     'wind': current_weather['Wind_Gust_Speed'],
#                     "pressure": current_weather["pressure"],
#                     "time1": time1, "time2": time2, "time3": time3, "time4": time4, "time5": time5,
#                     "temp1": f"{round(temp1,1)}", "temp2": f"{round(temp2,2)}", "temp3": f"{round(temp3,3)}", "temp4": f"{round(temp4,4)}", "temp5": f"{round(temp5,5)}",
#                     "hum1": f"{round(hum1,1)}", "hum2": f"{round(hum2,2)}", "hum3": f"{round(hum3,3)}", "hum4": f"{round(hum4,4)}", "hum5": f"{round(hum5,5)}",
#                 }
#             except requests.exceptions.RequestException as e:
#                 context['error'] = f"Could not connect to the weather API: {e}"
#             except KeyError as e:
#                 context['error'] = f"Error parsing weather data: {e}"
#             except FileNotFoundError:
#                 context['error'] = "Could not find the historical weather data file."
#             except Exception as e:
#                 context['error'] = f"An unexpected error occurred: {e}"

#     return render(request, 'weather.html', context)