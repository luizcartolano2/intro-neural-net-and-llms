import os

from matplotlib import pyplot as plt

from utils.generate_data import generate_regression_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

if __name__ == "__main__":
    # Criar pasta de saída caso não exista
    output_dir = "results/project4"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "mlp_results.txt")
    plot_file = os.path.join(output_dir, "mlp_surface_3d.png")

    # Gerar os dados
    data = generate_regression_data()
    X = data[['Receita por Dólar', 'Número de Escritórios']].values
    y = data[['Margem de Lucro']].values.ravel()

    # Dividir treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Criar MLP
    mlp = MLPRegressor(
        hidden_layer_sizes=(10, 5),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42,
        verbose=False
    )

    # Treinar
    mlp.fit(X_train_scaled, y_train)

    # Predições
    y_pred = mlp.predict(X_test_scaled)

    # Métricas
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Salvar resultados
    with open(output_file, "w") as f:
        f.write("MLP Regression Results\n")
        f.write("=====================\n\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"R²: {r2:.4f}\n")
        f.write(f"MAPE (%): {mape:.2f}\n\n")

        f.write("Predições vs Valores Reais (conjunto de teste):\n")
        for real, pred in zip(y_test, y_pred):
            f.write(f"{real:.4f} -> {pred:.4f}\n")

        f.write("\nPesos das camadas:\n")
        for i, layer_weights in enumerate(mlp.coefs_):
            f.write(f"Layer {i + 1} weights:\n{layer_weights}\n\n")

        f.write("Bias das camadas:\n")
        for i, layer_bias in enumerate(mlp.intercepts_):
            f.write(f"Layer {i + 1} bias:\n{layer_bias}\n\n")

    print(f"Todos os resultados foram salvos em: {output_file}")

    # --- Gerar scatter 3D ---
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot dos pontos reais
    ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='blue', label='Dados reais')

    # Malha para superfície aproximada pela MLP
    x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
    x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
    X_grid = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])
    X_grid_scaled = scaler.transform(X_grid)
    Z_grid = mlp.predict(X_grid_scaled).reshape(X1_grid.shape)

    ax.plot_surface(X1_grid, X2_grid, Z_grid, color='orange', alpha=0.5, label='Superfície MLP')

    ax.set_xlabel('Receita por Dólar')
    ax.set_ylabel('Número de Escritórios')
    ax.set_zlabel('Margem de Lucro')
    ax.set_title('MLP: Superfície Ajustada vs Dados Reais')

    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.show()

    print(f"Figura 3D salva em: {plot_file}")
