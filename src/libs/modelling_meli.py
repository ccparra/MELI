import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from typing import Any,List,Dict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import AgglomerativeClustering


class DataModelling:
    """
    Clase para realizar clustering de sellers.
    """

    @staticmethod
    def plot_dendrogram(model:Any, **kwargs):
        """
        Plotea un dendrograma utilizando el modelo de clustering jerárquico.

        Args:
            model: El modelo de clustering jerárquico.
            **kwargs: Argumentos adicionales a pasar a la función dendrogram.
        """
        # Crear matriz de enlace y luego plotear el dendrograma

        # Crear recuento de muestras bajo cada nodo
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # nodo hoja
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        # Plotear el dendrograma correspondiente
        dendrogram(linkage_matrix, **kwargs)

    @staticmethod
    def find_consecutive_percentage_change(values:List, threshold:float)->int:
        """
        Encuentra la posición del primer cambio porcentual consecutivo que sea inferior al umbral dado.

        Args:
            values (list): La lista de valores.
            threshold (float): El umbral para el cambio porcentual.

        Returns:
            int or None: La posición del cambio porcentual consecutivo o None si no se encuentra.
        """
        for i in range(0, len(values) - 1):
            current_value = values[i]
            next_value = values[i + 1]
            percentage_change = abs((next_value - current_value) / current_value) * 100
            if percentage_change < threshold:
                return i - 1
        return None

    @staticmethod
    def kmeans_clustering_analysis(df_scaled: pd.DataFrame, range_n_clusters: range) -> Dict:
        """
        Realiza análisis de clustering utilizando el algoritmo K-means.

        Args:
            df_scaled (pd.DataFrame): El dataframe de datos escalados.
            range_n_clusters (range): El rango de número de clusters a probar.

        Returns:
            List[int]: Lista con los resultados de los análisis.
        """
        inertias = []
        sill_values = []
        db_values = []
        ch_values = []

        for n_clusters in range_n_clusters:
            modelo_kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=20,
                random_state=123
            )
            cluster_labels = modelo_kmeans.fit_predict(df_scaled)
            silhouette_avg = silhouette_score(df_scaled, cluster_labels)
            davies_score = davies_bouldin_score(df_scaled, cluster_labels)
            calinski_score = calinski_harabasz_score(df_scaled, cluster_labels)
            inertias.append(modelo_kmeans.inertia_)
            sill_values.append(silhouette_avg)
            db_values.append(davies_score)
            ch_values.append(calinski_score)

        fig, (ax11, ax12, ax13, ax14) = plt.subplots(1, 4, figsize=(20, 7))
        ax11.plot(range_n_clusters, inertias, marker='o')
        ax11.set_title("Evolución de la varianza intra-cluster total")
        ax11.set_xlabel('Número clusters')
        ax11.set_ylabel('Intra-cluster (inertia)')
        ax12.plot(range_n_clusters, sill_values, marker='o')
        ax12.set_title("Evolución de media de los índices silhouette")
        ax12.set_xlabel('Número clusters')
        ax12.set_ylabel('Media índices silhouette')
        ax13.plot(range_n_clusters, db_values, marker='o')
        ax13.set_title("Evolución Davies-Bouldin Score")
        ax13.set_xlabel('Número clusters')
        ax13.set_ylabel('Davies-Bouldin')
        ax14.plot(range_n_clusters, ch_values, marker='o')
        ax14.set_title("Evolución Calinski-Harabasz Score")
        ax14.set_xlabel('Número clusters')
        ax14.set_ylabel('Calinski-Harabasz')

        k_inertias = DataModelling.find_consecutive_percentage_change(inertias, 10)
        k_sill_values = sill_values.index(max(sill_values))
        k_db_values = db_values.index(min(db_values))
        k_ch_values = ch_values.index(max(ch_values))

        results = {
            'elbow':k_inertias + 2,
            'silhoutte':k_sill_values + 2,
            'similitud_ Davies-Bouldin':k_db_values + 2,
            'Cohesion_calinski-harabasz':k_ch_values + 2
        }

        return results

    @staticmethod
    def hierarchical_clustering_analysis(df_scaled: pd.DataFrame, range_n_clusters: range) -> Dict:
        """
        Realiza análisis de clustering utilizando el algoritmo jerarquico.

        Args:
            df_scaled (pd.DataFrame): El dataframe de datos escalados.
            range_n_clusters (range): El rango de número de clusters a probar.

        Returns:
            List[int]: Lista con los resultados de los análisis.
        """ 
        sill_values = []
        db_values=[]
        ch_values=[]

        for n_clusters in range_n_clusters:
            modelo = AgglomerativeClustering(
                            affinity   = 'euclidean',
                            linkage    = 'ward',
                            n_clusters = n_clusters
                    )

            cluster_labels = modelo.fit_predict(df_scaled)
            silhouette_avg = silhouette_score(df_scaled, cluster_labels)
            davies_score= davies_bouldin_score(df_scaled,cluster_labels)
            calinski_score= calinski_harabasz_score(df_scaled,cluster_labels)
            sill_values.append(silhouette_avg)                
            db_values.append(davies_score)
            ch_values.append(calinski_score)

        fig, (ax12,ax13,ax14) = plt.subplots(1, 3, figsize=(20, 7))
        ax12.plot(range_n_clusters, sill_values, marker='o')
        ax12.set_title("Evolución de media de los índices silhouette")
        ax12.set_xlabel('Número clusters')
        ax12.set_ylabel('Media índices silhouette')
        ax13.plot(range_n_clusters, db_values, marker='o')
        ax13.set_title("Evolución Davies-Bouldin Score")
        ax13.set_xlabel('Número clusters')
        ax13.set_ylabel('Davies-Bouldin')
        ax14.plot(range_n_clusters, ch_values, marker='o')
        ax14.set_title("Evolución Calinski-Harabasz Score")
        ax14.set_xlabel('Número clusters')
        ax14.set_ylabel('Calinski-Harabasz')

        k_sill_values = sill_values.index(max(sill_values))
        k_db_values = db_values.index(min(db_values))
        k_ch_values = ch_values.index(max(ch_values))

        results = {
            'silhoutte':k_sill_values + 2,
            'similitud_ Davies-Bouldin':k_db_values + 2,
            'Cohesion_calinski-harabasz':k_ch_values + 2
        }

        return results

    @staticmethod
    def cluster_characterization_numeric_var(df:pd.DataFrame,str_var: str) -> pd.DataFrame:
        """
        Calcula metricas por grupo para caracterizar los clusters.

        Args:
            df (pd.Dataframe): dataframe con variables a caracterizar.       
            str_var (str): Nombre de la variable a analizar.

        Returns:
            pd.DataFrame: DataFrame con las caracteristicas de la variable numerica por grupo.
        """
        grupo = df.groupby(['label'])[str_var]
        mean = grupo.mean()
        std = grupo.std()
        min = grupo.min()
        max = grupo.max()
        coef_var_g = std / mean
        coef_var_tot = df[str_var].std() / df[str_var].mean()
        result_df = pd.DataFrame({'grupo': coef_var_g.index,
                                'cv_grupo': coef_var_g,
                                'cv_total': coef_var_tot,
                                'dif%_cv': 100 * (coef_var_g / coef_var_tot - 1),
                                'min': min,
                                'max': max,
                                'mean': mean}).reset_index(drop=True)
        return result_df           