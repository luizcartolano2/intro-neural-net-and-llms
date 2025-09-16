import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import statsmodels.api as sm
import pandas as pd


def plot_3d_regression_plane(df, model, x1_col, x2_col, y_col,
                             grid_size=20, alpha=0.4, cmap=cm.viridis, figsize=(12, 8),
                             title='Plano de regressão ajustado', save_path=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot dos pontos reais
    ax.scatter(df[x1_col], df[x2_col], df[y_col], color='blue', label='Dados reais')

    # Malha para o plano ajustado
    x1_range = np.linspace(df[x1_col].min(), df[x1_col].max(), grid_size)
    x2_range = np.linspace(df[x2_col].min(), df[x2_col].max(), grid_size)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

    # Prever valores ajustados
    X_pred = sm.add_constant(pd.DataFrame({x1_col: x1_grid.ravel(), x2_col: x2_grid.ravel()}))
    y_pred = model.predict(X_pred).values.reshape(x1_grid.shape)

    # Plot do plano
    ax.plot_surface(x1_grid, x2_grid, y_pred, alpha=alpha, cmap=cmap)

    ax.set_xlabel(x1_col)
    ax.set_ylabel(x2_col)
    ax.set_zlabel(y_col)
    plt.title(title)
    plt.legend()

    # Salvar ou mostrar
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)  # fecha a figura para não mostrar na tela
    else:
        plt.show()
