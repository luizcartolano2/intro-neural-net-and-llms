import io
import pandas as pd


def generate_regression_data() -> pd.DataFrame:
    """ Generate a DataFrame with regression data.
    :return: a pandas DataFrame with regression data.
    """
    data = """
    1 & 3,92 & 7298 & 0,75 \\
    2 & 3,61 & 6855 & 0,71 \\
    3 & 3,66 & 6935 & 0,66 \\
    4 & 3,07 & 6506 & 0,61 \\
    5 & 3,36 & 6740 & 0,69 \\
    6 & 3,11 & 6402 & 0,72 \\
    7 & 3,12 & 6462 & 0,79 \\
    8 & 3,26 & 6430 & 0,74 \\
    9 & 3,42 & 6369 & 0,72 \\
    10 & 3,42 & 6356 & 0,82 \\
    11 & 3,51 & 6392 & 0,81 \\
    12 & 3,66 & 6798 & 0,73 \\
    13 & 3,66 & 6546 & 0,78 \\
    14 & 3,78 & 6672 & 0,84 \\
    15 & 3,82 & 6890 & 0,79 \\
    16 & 3,97 & 7115 & 0,70 \\
    17 & 4,07 & 7327 & 0,68 \\
    18 & 4,27 & 7542 & 0,65 \\
    19 & 4,41 & 7931 & 0,55 \\
    20 & 4,50 & 8097 & 0,63 \\
    21 & 4,70 & 8468 & 0,56 \\
    22 & 4,58 & 8717 & 0,47 \\
    23 & 4,69 & 8991 & 0,51 \\
    24 & 4,71 & 9179 & 0,41 \\
    25 & 4,78 & 9318 & 0,32 \\
    """

    # Clean the data
    cleaned_data = (
        data.replace(',', '.')
        .replace('&', ',')
        .replace('\\', '')
        .strip()
    )

    # Define columns names
    columns = ['Ano', 'Receita por Dólar', 'Número de Escritórios', 'Margem de Lucro']

    # Create a StringIO object
    data_io = io.StringIO(cleaned_data)

    # Read the data into a DataFrame
    df = pd.read_csv(data_io, names=columns, skipinitialspace=True)
    df = df.astype({
        'Ano': 'int',
        'Receita por Dólar': 'float',
        'Número de Escritórios': 'int',
        'Margem de Lucro': 'float'
    })

    return df
