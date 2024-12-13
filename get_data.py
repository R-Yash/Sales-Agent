import requests
import pandas as pd

prods = []
for i in range(1,4):
    res = requests.get(f'https://drsambunting.com/products.json?page={i}')
    prods += res.json()['products']

df = pd.DataFrame(prods)

df = df.drop(['images','options'],axis=1)

df.to_csv('products.csv')