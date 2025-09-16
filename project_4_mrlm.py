import json

import statsmodels.api as sm

from utils.analyse_regression import analyse_regression
from utils.generate_data import generate_regression_data
from utils.plot_regression import plot_3d_regression_plane

if __name__ == "__main__":
    data = generate_regression_data()

    X = data[['Receita por Dólar', 'Número de Escritórios']]
    X = sm.add_constant(X)
    model = sm.OLS(data['Margem de Lucro'], X)
    results = model.fit()
    stats_results = analyse_regression(results, data[['Receita por Dólar', 'Número de Escritórios']])

    with open("results/project4/stats_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(results.summary()))

    with open("results/project4/regression_analysis.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(stats_results, indent=4))

    plot_3d_regression_plane(data, results, 'Receita por Dólar', 'Número de Escritórios', 'Margem de Lucro',
                             save_path='results/project4/regression_plane.png')
