import json
from typing import List,Dict
import pandas as pd

def filter_list_of_dicts_by_string_in_key(list_of_dicts: List[dict], key_name: str, string_value: str) -> List[dict]:
    """
    Filtra una lista de diccionarios por un valor de cadena en una clave específica.

    Args:
        list_of_dicts (List[dict]): Lista de diccionarios a filtrar.
        key_name (str): Nombre de la clave en la cual buscar el valor de cadena.
        string_value (str): Valor de cadena a buscar.

    Returns:
        List[dict]: Lista de diccionarios que contienen el valor de cadena en la clave especificada.
    """
    result = []
    for dict in list_of_dicts:
        if key_name in dict and isinstance(dict[key_name], str) and string_value in dict[key_name]:
            result.append(dict)
    return result

def get_list_values_by_key_of_dict(dictionary_list: List[dict], key_name: str) -> List:
    """
    Obtiene una lista de valores de una clave específica de una lista de diccionarios.

    Args:
        dictionary_list (List[dict]): Lista de diccionarios.
        key_name (str): Nombre de la clave para obtener los valores.

    Returns:
        List: Lista de valores correspondientes a la clave especificada.
    """
    values = []
    for dictionary in dictionary_list:
        if key_name in dictionary:
            values.append(dictionary[key_name])
    return values

def expand_dict_column(df, column_name)->pd.DataFrame:
    """
    Identifica si una columna tiene como valores diccionarios y convierte los items de esos diccionarios en nuevas columnas del DataFrame original.

    Args:
        df (pandas.DataFrame): El DataFrame original.
        column_name (str): El nombre de la columna a expandir.

    Returns:
        pandas.DataFrame: El DataFrame resultante con las columnas expandidas en caso de que la columna contenga diccionarios.
    """
    # Verifica si la columna contiene diccionarios
    if df[column_name].apply(lambda x: isinstance(x, dict)).all():
        # Expande los diccionarios en columnas
        expanded_df = pd.json_normalize(df[column_name])
        # Combina el DataFrame original con el DataFrame expandido
        df = pd.concat([df, expanded_df], axis=1)
        # Elimina la columna original de diccionarios
        df.drop(column_name, axis=1, inplace=True)

    return df