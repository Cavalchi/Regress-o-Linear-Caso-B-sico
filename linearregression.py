import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

# Diretório de saída
exitdir = 'Resultados'
os.makedirs(exitdir, exist_ok=True)

# Ler os dados da tabela csv
data = pd.read_csv('GlobalCO2Emissions.csv')

# Extrair year e value
year = data['Year'].values
value = data['Emissions'].values

# Função para fazer a previsão usando regressão linear
def predict_future(values, years_ahead):
    if len(values) < 2:
        raise ValueError("São necessários pelo menos dois valores para fazer esta previsão.")
    
    X = np.array(range(len(values))).reshape(-1, 1)
    y = np.array(values)
    model = LinearRegression().fit(X, y)
    future_X = np.array([[len(values) + years_ahead]])
    return round(model.predict(future_X)[0], 2)

# Prever value futuros de 10 em 10 anos até 100anos  no futuro
year_future = []
value_future = []
for i in range(10, 101, 10):
    year_future.append(year[-1] + i)
    value_future.append(predict_future(value, i))

# Criar um df com os resultados
resulte = pd.DataFrame({
    'Year': year_future,
    'Emissions': value_future
})

# Salvar o resultado em um  CSV
resulte.to_csv(os.path.join(exitdir, 'previsao_co2.csv'), index=False)