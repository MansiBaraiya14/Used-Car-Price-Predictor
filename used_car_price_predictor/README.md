# Used Car Price Predictor

A web app that predicts the price of a used car in India and shows what affects used-car prices.
I built it for my Python course using the CarDekho dataset from Kaggle.

## Why I chose this project
Buying or selling a used car is hard because it's difficult to know what a fair price is.
I wanted to see which things (age, kilometres driven, fuel type, brand and so on) change the price
the most, and whether a model could estimate the price from them.

## Dataset
- Source: CarDekho used car dataset from Kaggle
- Columns: brand, model, vehicle age, km driven, seller type, fuel type, transmission, mileage,
  engine, max power, seats and selling price (in ₹)
- The dataset is not included in this repository. Download it from Kaggle and put the CSV in `data/raw/`.

## What the app does
- Predict price:choose a car's details and get an estimated price.
- Explore data: charts showing how price changes with age, kilometres, brand, fuel type and transmission.
- Model performance: how accurate the model is on cars it hasn't seen.
- Data & limitations:the cleaned data and a summary of what was removed.

## How I built it
1. Cleaned the data: removed duplicates and rows with missing or impossible values.
2. Explored it with charts to see which features matter.
3. Trained two models, Ridge Regression (simple baseline) and Random Forest, using scikit-learn.
4. Predicted the log of the price, because prices are very uneven (a few cars cost far more than most).
5. Compared the models on a test set of cars the model had not seen.
6. Built the app with Streamlit and charts with Plotly.

## How to run
```
pip install -r requirements.txt
python -m streamlit run app.py
```
To run the tests: `python -m pytest`

## Project structure
- `app.py`: the Streamlit app
- `src/data.py`: loading and cleaning the data
- `src/model.py`: training and evaluating the models
- `tests/`: automated tests

## Limitations
- The data doesn't include the car's condition, accident history or city, which also affect price.
- The prices come from one time period and may not match today's market.
- The model is less reliable for rare cars that appear only a few times in the data.
- The results show patterns in the data, not proof of what causes a price change.

## What I learned
- Cleaning data takes a big share of the work. The raw dataset had duplicates and values
  that don't make sense, so I had to clean it before any model could use it.
  Learning to do this with pandas was one of the most useful parts of the project.

- A prediction depends on all the details together. For a 10-year-old diesel Maruti Swift Dzire
  with about 59,000 km driven, my app predicted about ₹3.27 lakh. The typical listed price for the
  Swift Dzire across all ages in the data is ₹5.21 lakh. The prediction is lower because that
  median includes newer cars, and age and kilometres driven bring the price down.
  This showed me that one average price can be misleading if you ignore the car's details.

- Machine learning has limits. The data doesn't include the car's condition, accident history or
  city, so the app gives an estimate, not an exact valuation.

- Building an app makes a project easier to understand. Using Streamlit let me turn my
  code into something anyone can use by choosing options instead of reading code.

- What I would improve next: I would like to improve my project by adding features like city or car condition , damage history of the car, trying other models or deploying the app online.

## Note
I used an AI assistant to help build this project.




