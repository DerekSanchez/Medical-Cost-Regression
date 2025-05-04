import pandas as pd
import matplotlib.pyplot as plt
from src.config import paths
import os
from datetime import datetime
from IPython.display import display

# loads data from a path
def load_data(path):
    """
    Loads a dataset from the specified path
    
    Parameters:
        path (str): path to the file
    
    Returns:
        pd.DataFrame: pandas dataframe loaded to python enviroment
    """
    return pd.read_csv(path)

# saves data to a path
def save_data(df, path):
    """
    Saves a dataset to the specified path
    
    Parameters:
        df (pd.DataFrame): dataset to save
        path (str): path to save to
    """
    df.to_csv(path, index = False)
    print(f"Data saved to {path}")

def save_plot(fig, filename):
    """
    Saves a plot to the specified path
    
    Parameters:
        fig (matplotlib.figure.figure): Figure to save
        filename (str): Path and file name to save the plot to
    """
    fig.savefig(filename)
    print(f"Gráfico guardado en: {filename}")

# summarize information
def summary_info(df):
    """
    Clean Summary of df containing:
    - Column name
    - Data type
    - Non-null percentage values
    - Null percentage values
    - Unique values

    Parameters:
        df (pd.DataFrame): DataFrame to analyze

    Returns:
        pd.DataFrame.style: Summarized DataFrame with styles applied
    """
    # create summary
    summary = pd.DataFrame({
        "Column": df.columns,
        "Non-Null Count": df.notnull().sum(),
        "Missing %": df.isnull().mean() * 100,
        "Unique Values": df.nunique(),
        "Data Type": df.dtypes
    })

    # sort by missing values %
    summary = summary.sort_values(by="Missing %", ascending=False).reset_index(drop=True)

    # format numeric columns
    summary["Missing %"] = summary["Missing %"].round(2)
    summary["Non-Null Count"] = summary["Non-Null Count"].apply(lambda x: f"{x:,.0f}")
    summary["Unique Values"] = summary["Unique Values"].apply(lambda x: f"{x:,.0f}")

    # apply styles
    styled_summary = (
        summary.style
        .set_properties(**{'text-align': 'center'})  # center columns
        .hide(axis = 'index')  # hide index
        .set_table_styles([  # center headers
            {"selector": "th", "props": [("text-align", "center")]}
        ])
    )

    # show styled table
    display(styled_summary)
    
# Analyze uniqueness of categorical features
def uniqueness_categorical_columns(df, max_categories = 10):
    """
    Analyzes categorical columns showing:
    1. number of unique values
    2. porcentual distribution of categories
    
    Parameters:
        df (pd.DataFrame): DataFrame to analyze
    """
    categorical_cols = df.select_dtypes(include = 'object').columns
    
    for col in categorical_cols:
        print(f'--- Column: {col} ---')
        print(f'Unique values: {df[col].nunique()}')
        
        distribution = df[col].value_counts(normalize=True) * 100
        distribution_top = distribution.head(max_categories)
        
        # Create a table for clean visualization
        table = pd.DataFrame({
            'Category': distribution_top.index,
            'Percentage': distribution_top.values
        })
        
        # Apply format and style
        
        styled_table = (
            table.style
            .format({'Percentage': "{:.2f}%"}) # percentage format
            .set_properties(**{'text-align': 'center'}) # center columns
            .hide(axis = "index") # hidde index for cleanliness
        )
        
        display(styled_table)
        
        if len(distribution) > max_categories:
            print(f'... showing the top {max_categories} most common values')
        print("\n")

# Pivotings NAs
def missing_values_by_pivot(df, pivot_col=None, return_type="count", percentage_base="total", show_all=True):
    """
    Muestra el número o porcentaje de valores faltantes en cada columna,
    opcionalmente agrupado por las categorías de una columna pivote.

    Parameters:
        df (pd.DataFrame): DataFrame a analizar.
        pivot_col (str, optional): Columna para pivotear. Si es None, se calcula sin pivotear.
        return_type (str): "count" para número de NAs o "percentage" para porcentaje.
        percentage_base (str): Base para calcular porcentaje (aplica solo a "percentage"):
            - "total": Con respecto al total de observaciones del DataFrame.
            - "column": Con respecto al total de observaciones de cada columna dentro de cada categoría del pivote.
            - "row": Con respecto al total de observaciones por fila dentro de cada categoría del pivote.
        show_all (bool): Si True, muestra todas las columnas. Si False, solo columnas con NAs.

    Returns:
        pd.DataFrame: Tabla resumen con los NAs (número o porcentaje) por columna.
    """
    if pivot_col and pivot_col not in df.columns:
        raise ValueError(f"La columna '{pivot_col}' no está en el DataFrame.")
    if return_type not in ["count", "percentage"]:
        raise ValueError("El parámetro return_type debe ser 'count' o 'percentage'.")
    if return_type == "percentage" and percentage_base not in ["total", "column", "row"]:
        raise ValueError("El parámetro percentage_base debe ser 'total', 'column', o 'row'.")

    # Si no se especifica pivot_col, calcular totales sin pivotear
    if pivot_col is None:
        if return_type == "count":
            result = df.isnull().sum().to_frame(name="Missing Values")
        elif return_type == "percentage":
            if percentage_base == "total":
                total = df.shape[0]
                result = (df.isnull().sum() / total * 100).to_frame(name="Missing %")
            elif percentage_base == "column":
                result = (df.isnull().mean() * 100).to_frame(name="Missing %")
        if not show_all:
            result = result[result.iloc[:, 0] > 0]
        return result

    # Si se especifica pivot_col, calcular por categorías
    grouped = df.groupby(pivot_col)
    
    if return_type == "count":
        result = grouped.apply(lambda group: group.isnull().sum()).T
    elif return_type == "percentage":
        if percentage_base == "total":
            total = df.shape[0]
            result = grouped.apply(lambda group: group.isnull().sum() / total * 100).T
        elif percentage_base == "column":
            result = grouped.apply(lambda group: group.isnull().mean() * 100).T
        elif percentage_base == "row":
            result = grouped.apply(lambda group: group.isnull().sum() / len(group) * 100).T

    # Filtrar columnas sin NAs si show_all=False
    if not show_all:
        result = result.loc[result.sum(axis=1) > 0]

    return result

# Missing Values by date
def missing_values_by_date_pivot(df, date_col, return_type="count", percentage_base="total", num_dates=3, show_all=True):
    """
    Analiza valores faltantes agrupados por fechas en una columna de tipo datetime,
    con la opción de limitar la cantidad de fechas visibles desde la más reciente.

    Parameters:
        df (pd.DataFrame): DataFrame a analizar.
        date_col (str): Columna de tipo fecha para pivotear.
        return_type (str): "count" para número de NAs o "percentage" para porcentaje.
        percentage_base (str): Base para calcular porcentaje (aplica solo a "percentage"):
            - "total": Con respecto al total de observaciones del DataFrame.
            - "column": Con respecto al total de observaciones de cada columna dentro de cada categoría del pivote.
            - "row": Con respecto al total de observaciones por fila dentro de cada categoría del pivote.
        num_dates (int): Número de fechas a mostrar (más recientes hacia atrás).
        show_all (bool): Si True, muestra todas las columnas. Si False, solo columnas con NAs.

    Returns:
        pd.DataFrame: Tabla resumen con los NAs (número o porcentaje) por fecha.
    """
    if date_col not in df.columns:
        raise ValueError(f"La columna '{date_col}' no está en el DataFrame.")
    if return_type not in ["count", "percentage"]:
        raise ValueError("El parámetro return_type debe ser 'count' o 'percentage'.")
    if return_type == "percentage" and percentage_base not in ["total", "column", "row"]:
        raise ValueError("El parámetro percentage_base debe ser 'total', 'column', o 'row'.")

    # Asegurar que la columna es de tipo datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Ordenar por fecha de más antigua a más reciente
    df = df.sort_values(by=date_col)

    # Agrupar por la columna de fecha
    grouped = df.groupby(date_col)

    if return_type == "count":
        result = grouped.apply(lambda group: group.isnull().sum()).T
    elif return_type == "percentage":
        if percentage_base == "total":
            total = df.shape[0]
            result = grouped.apply(lambda group: group.isnull().sum() / total * 100).T
        elif percentage_base == "column":
            result = grouped.apply(lambda group: group.isnull().mean() * 100).T
        elif percentage_base == "row":
            result = grouped.apply(lambda group: group.isnull().sum() / len(group) * 100).T

    # Filtrar columnas sin NAs si show_all=False
    if not show_all:
        result = result.loc[result.sum(axis=1) > 0]

    # Seleccionar las últimas num_dates fechas
    if num_dates:
        result = result.iloc[:, -num_dates:]

    return result

# Formating de celdas
def format_cell(val):
    """
    Formatea valores para mostrar en tablas:
    - Enteros con separadores de miles.
    - Flotantes con dos decimales.
    - Valores nulos como "N/A".
    
    Parameters:
        val: Cualquier valor de entrada.

    Returns:
        str: Representación formateada del valor.
    """
    import pandas as pd
    
    if pd.isnull(val):  # Manejo de valores nulos
        return "N/A"
    elif isinstance(val, (int, float)):  # Manejo de valores numéricos
        if isinstance(val, int) or val == int(val):  # Enteros
            return f"{int(val):,}"
        else:  # Flotantes
            return f"{val:,.2f}"
    return str(val)  # Para otros tipos, convertir a cadena directamente

# Función para centrar
def center_align(val):
    return 'text-align: center;'

def write_log(message, log_path = paths['logs']):
    """
    Writes messages in a log file
    
    Parameters:
        - message (str): Message to register
        - log_path(str): Path of log file
    """
    
    # check if directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok = True)
    
    # write in the file
    with open(log_path, 'a') as log_file:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f'[{timestamp}] {message}\n')
    
    print(f'Log has been registered: {message}')