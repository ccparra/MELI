from typing import List, Dict
import pandas as pd
import numpy as np
import requests
import datetime as dt

class data_handling:
    def __init__(self):
        pass

    @staticmethod
    def filter_list_of_dicts_by_string_in_key(list_of_dicts: List[Dict], key_name: str, string_value: str) -> List[Dict]:
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

    @staticmethod
    def get_list_values_by_key_of_dict(dictionary_list: List[Dict], key_name: str) -> List:
        """
        Obtiene una lista de valores de una clave específica de una lista de diccionarios.

        Args:
            dictionary_list (List[dict]): Lista de diccionarios.
            key_name (str): Nombre de la clave para obtener los valores.

        Returns:
            List: Lista de valores correspondientes a la clave especificada.
        """
        #gebera lista vacia de almacenamiento
        values = []
        #itera sobre los diccionarios de la lista
        for dictionary in dictionary_list:
            #verifica si key_name pertenece al diccionario
            if key_name in dictionary:
                #adiciona valores
                values.append(dictionary[key_name])
        return values

    @staticmethod
    def expand_dict_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
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

    @staticmethod
    def summarise_categoric_variable(df: pd.DataFrame, row_name: str, column_name: str) -> pd.DataFrame:
        """
        Genera una tabla cruzada personalizada a partir de un DataFrame.

        Parameters:
        - df (pandas.DataFrame): El DataFrame que contiene los datos.
        - column_name (str): El nombre de la columna que se utilizará como filas en la tabla cruzada.
        - row_name (str): El nombre de la columna que se utilizará como columnas en la tabla cruzada.

        Returns:
        - DataFrame: La tabla cruzada resultante con el índice reseteado y el nombre de columna renombrado.
        """
        #Genera Tabla Cruzada
        tabla_cruzada = pd.crosstab(df[row_name], df[column_name], normalize='index', dropna=False)
        #renombra columnas
        tabla_cruzada = tabla_cruzada.reset_index(drop=False).rename_axis(" ", axis='columns')
        return tabla_cruzada

    @staticmethod
    def get_items_data(tot_items_cat: int, cat_id: str, country_id: str) -> pd.DataFrame:
        """
        Obtiene datos de items a partir de una categoría, la cantidad total de items y el ID del país.

        Args:
            tot_items_cat (int): Cantidad total de items de la categoría.
            cat_id (str): ID de la categoría.
            country_id (str): ID del país.

        Returns:
            pandas.DataFrame: DataFrame con los datos de los items obtenidos.
        """
        #genera rango de offsets values
        offset = range(0, (tot_items_cat // 50) * 50 + 1, 50)
        #genera dataframe vacio de almacenamiento
        df_items = pd.DataFrame()

         #itera sobre los offsets   
        for i in offset:
            #realiza request para obtener items_data
            url = f'https://api.mercadolibre.com/sites/{country_id}/search?category={cat_id}&offset={i}'
            response = requests.get(url)
            items_tmp = response.json()
            #apila respuestas de cada request
            df_items = pd.concat([df_items, pd.DataFrame(items_tmp.get('results'))], axis=0, ignore_index=True)
        #renombra y transforma columnas
        df_items = df_items.rename(columns={'id': 'item_id'})
        df_items = data_handling.expand_dict_column(df_items, 'seller')
        df_items = data_handling.expand_dict_column(df_items, 'installments')
        df_items = df_items.rename(columns={'id': 'seller_id'})

        return df_items

    @staticmethod
    def transform_df_items(df_items: pd.DataFrame) -> pd.DataFrame:
        """
        Genera una tabla cruzada personalizada a partir de un DataFrame.

        Parameters:
        - df_items (pandas.DataFrame): El DataFrame que contiene los datos de items.
        
        Returns:
        - DataFrame: dataframe con la información de seller sumarizada
        """
        df = (
            df_items
            .assign(
                discount=lambda df: np.where(((df['price'] < df['original_price']) & (df.original_price.notna())), 100 * (1 - df['price'] / df['original_price']), 0)
            )
            .assign(
                registration_date=lambda df: pd.to_datetime(df['registration_date']).dt.strftime('%Y-%m-%d'),
                years_antiquity=lambda df: (dt.datetime.today() - pd.to_datetime(df['registration_date'])).dt.days / 365.25
            )
            .groupby(["seller_id", "seller_reputation.power_seller_status", 'seller_reputation.level_id', 
                      'registration_date', 'years_antiquity', 'seller_reputation.metrics.sales.period'], dropna=False)
            .agg(
                n_items=('item_id', 'count'),
                n_subcategories=("category_id", 'nunique'),
                sold_items=('sold_quantity', 'sum'),
                mean_intallments=('quantity', 'mean'),
                q10_discount=('discount', lambda x: x.quantile(0.1)),
                q50_discount=('discount', lambda x: x.quantile(0.5)),
                q90_discount=('discount', lambda x: x.quantile(0.9)),
                total_transactions=('seller_reputation.transactions.total', 'sum'),
                completed_transactions=('seller_reputation.transactions.completed', 'sum'),
                canceled_transactions=('seller_reputation.transactions.canceled', 'sum'),
                negative_rating=('seller_reputation.transactions.ratings.negative', 'mean'),
                neutral_rating=('seller_reputation.transactions.ratings.neutral', 'mean'),
                positive_rating=('seller_reputation.transactions.ratings.positive', 'mean'),
                total_claims=('seller_reputation.metrics.claims.value', 'sum'),
                total_delayed_shipments=('seller_reputation.metrics.delayed_handling_time.value', 'sum'),
                total_cancellations=('seller_reputation.metrics.cancellations.value', 'sum')
            )
            .reset_index(drop=False)
            .assign(**{'seller_reputation.power_seller_status': lambda df: np.where(df['seller_reputation.power_seller_status'].notna(), df['seller_reputation.power_seller_status'], 'Not_medal')})
            .assign(**{'seller_reputation.level_id': lambda df: np.where(df['seller_reputation.level_id'].notna(), df['seller_reputation.level_id'], '0_Not_type')})
            .merge(data_handling.summarise_categoric_variable(df_items, row_name='seller_id', column_name='listing_type_id'), how='left', on='seller_id')
            .reset_index(drop=True)
            .drop(columns=['registration_date'])
        )

        df = df.rename(columns=dict(zip([c for c in df.columns if '.' in c], [c.replace('.', '+') for c in df.columns if '.' in c])))

        dummies = pd.get_dummies(df.select_dtypes(include=['object']), prefix_sep='-', dtype='float64').assign(seller_id=df['seller_id'])
        df = df.merge(dummies, how='left', on='seller_id').drop(columns=df.select_dtypes(include=['object']).columns)

        return df
    