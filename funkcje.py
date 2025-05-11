import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import re

# Biblioteki do rysowania Confidence level Elipses oraz ConvexHull
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.spatial import ConvexHull

# Biblioteki do tworzeniu modeli redukcji wymiarowości
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.manifold import MDS
import umap
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

# Biblioteki do tworzenia modeli technik nadzorowanych i metryki
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

# Funkcja do zmiany nazw kolumn Unnamed
def rename_unnamed_columns(df, start):
    new_columns = []
    for col in df.columns:
        if 'Unnamed:' in col:
            new_name = f'Unnamed:_{start}'
            start += 2
            new_columns.append(new_name)
        else:
            new_columns.append(col)
    df.columns = new_columns
    return df


# Funkcja do zamiany na pełne nazwy, zachowując liczby
def replace_name(name, dictionary):
    # Znajdujemy liczbę na końcu (jeśli występuje) i przechowujemy ją
    match = re.search(r'(\d)$', name)
    number = match.group(0) if match else ""
    
    # Usuwamy liczbę z nazwy, aby dokonać zamian tekstowych
    base_name = name[:-1] if number else name
    
    # Dokonujemy zamiany na podstawie słownika
    for old, new in dictionary.items():
        base_name = base_name.replace(old, new)
    
    # Dodajemy liczbę z powrotem do końca nazwy, jeśli była
    return base_name + number


# Funkcja do tworzenia nowych kolumn bazujac na nazwie próbki
def split_column_names(df):
    # Zaktualizowane wyrażenie regularne
    pattern = r'(\w+(?:_\w+)*?)_(początek|otwarty|zamknięty)([1-5]?)(?:_(\d+t\d*\d*))?_SYN_(30|40|50|60|70|80|90|100)'

    # Lista nowych wierszy
    new_data = []

    for col in df.columns:
        # Dopasowanie wyrażenia do nazwy kolumny
        match = re.match(pattern, col)
        
        # Wyodrębnienie grup
        if match:
            nazwa = match.group(1)  # [Nazwa]
            status = match.group(2)  # [Status]
            sample_number = match.group(3) if match.group(3) else '0'  # [SampleNumber]
            czas = match.group(4) if match.group(4) else 'początek'  # [Czas], jeśli jest
            dl = match.group(5)  # [dL]
            
            # Wyodrębnienie numeru próbki z [Czas]
            if re.match(r'\d+t\d*\d*', czas):
                sample_number_in_time = re.match(r'\d+t(\d*\d*)', czas).group(1)
                sample_number = sample_number_in_time if sample_number_in_time else '0'
                
                # Usuń numer próbki z [Czas]
                czas = re.sub(r'(\d+t)\d*', r'\1', czas)
            else:
                sample_number_in_time = '0'
            
            # Dodanie nowych kolumn do DataFrame
            new_data.append([nazwa, status, sample_number, czas, dl])

    # Przekształcenie listy w nowy DataFrame
    new_df = pd.DataFrame(new_data, columns=['Nazwa', 'Status', 'Numer', 'Czas', 'dL'])
    
    # Tworzymy DataFrame z wartościami od 240 do 600 (włącznie)
    additional_columns = pd.DataFrame(np.nan, index=new_df.index, columns=[str(i) for i in range(240, 601)])
    
    # Łączenie nowego DataFrame z dodatkowymi kolumnami
    final_df = pd.concat([new_df, additional_columns], axis=1)
    
    return final_df


# Funkcja do rysowania elips poziomów ufności
def confidence_ellipse(x, y, ax, n_std=1.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    
    # Using a special case to obtain the eigenvalues of this two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)
    
    # Calculating the standard deviation of x from the squareroot of the variance and multiplying with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    
    # Calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# Funkcja do tworzenia modeli
def create_model(model_name, data, labels, columns, 
                 n_components=2, y=None, **kwargs):
    
    # Tworzenie modelu z zadaną liczbą komponentów
    model = model_name(n_components=n_components, **kwargs)
    
    # Przekształcanie danych za pomocą modelu
    if y is not None: 
        data_transformed = model.fit_transform(data, y)
    else:
        data_transformed = model.fit_transform(data)
    
    # Tworzenie DataFrame z głównymi komponentami
    model_df = pd.DataFrame(data=data_transformed, columns=columns)

    # Dodanie etykiet do DataFrame
    model_df = pd.concat([model_df, labels.reset_index(drop=True)], axis=1)
    
    return model_df, model


# Funkcja do rysowania wykresów
def draw(model_df, discriminant_cols, group_column='dL', hue='Nazwa', size='Czas', style='Status',
         use_hull=False, use_confidence_ellipse=False, **kwargs):
    
    for dL in model_df[group_column].unique():
        plt.figure(figsize=(10, 6))
        subset = model_df[model_df[group_column] == dL]
        
        # Mapa stylów dla 'Status' 
        # style_dict = {'początek': 'o', 'otwarty': 'x', 'zamknięty': 's'}
        
        # Tworzenie wykresu scatterplot
        ax = sns.scatterplot(x=discriminant_cols[0], 
                             y=discriminant_cols[1], 
                             hue=hue, 
                             size=size, 
                             style=style, 
                             data=subset, 
                             palette='cubehelix', 
                             sizes=(200, 20), 
                             legend='brief', 
                             alpha=0.7)
        
        # Dodawanie Convex Hull, jeśli jest włączony
        if use_hull:
            for key, group in subset.groupby(hue):
                points = group[discriminant_cols].values
                if len(points) >= 3:  # ConvexHull wymaga co najmniej 3 punktów
                    hull = ConvexHull(points)
                    for simplex in hull.simplices:
                        ax.plot(points[simplex, 0], points[simplex, 1], 'k-', lw=0.5)
                        
        # Dodanie elips poziomów ufności, jeśli są włączone
        if use_confidence_ellipse:
            for key, group in subset.groupby(hue):
                confidence_ellipse(
                    group[discriminant_cols[0]], group[discriminant_cols[1]], 
                    ax, edgecolor='red', **kwargs)
                        
        # Ustawienia dla wykresu
        plt.title(f'Dyskryminanty dla {group_column} = {dL}')
        plt.xlabel(discriminant_cols[0])
        plt.ylabel(discriminant_cols[1])
        plt.legend(title='Legenda')
        plt.show()


# Funkcja do losowego podziału danych na zbiory treningowe i testowe
def split_data_randomly(data, y, n_train=4):
          
    # Unikalne wartości w kolumnie 'Numer'
    unique_numbers = data['Numer'].unique()
    
    # Losowy wybór numerów do zbiorów
    train_numbers = np.random.choice(unique_numbers, size=n_train, replace=False)
    test_numbers = [num 
                    for num in unique_numbers 
                    if num not in train_numbers
                    ]
    
    # Tworzenie masek dla zbiorów
    train_mask = data['Numer'].isin(train_numbers)
    test_mask = data['Numer'].isin(test_numbers)
    
    # Podział danych
    X = data.drop(columns=['Nazwa', 'Status', 'Czas', 'dL', 'Numer'])
    y = y
    
    # Podział zbiorów
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    return X_train, X_test, y_train, y_test


# Funkcja do skalowania danych
def scale_data(X_train, X_test):
    
    # Tworzenie skalara
    scaler = StandardScaler()
    
    # Skalowanie danych
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled
    

# Funkcja do tworzenia modelu nadzorowanego
def create_supervised_model(model_name, X_train, y_train, X_test, **kwargs):
    
    # Tworzenie modelu nadzorowanego
    model = model_name(**kwargs)
    
    # Trening modelu nadzorowanego
    model.fit(X_train, y_train)
    
    # Przewidywanie na zbiorze testowym
    y_pred = model.predict(X_test)
    
    return y_pred
    

# Funkcja do rysowania dendogramu
def draw_dendrogram_by_dl(data, x_cols, group_column, label_column, method='ward'):
    """
    Rysuje osobne dendrogramy dla każdej unikalnej wartości w kolumnie `group_column`.

    Args:
        data (pd.DataFrame): Dane wejściowe zawierające kolumny do klasteryzacji oraz grupowanie.
        x_cols (list): Lista kolumn, które będą użyte do klasteryzacji (np. ['PC1', 'PC2']).
        group_column (str): Nazwa kolumny do grupowania danych (np. 'dL').
        label_column (str): Kolumna używana jako etykiety w dendrogramach (np. 'Nazwa').
        method (str): Metoda linkage do klasteryzacji hierarchicznej (domyślnie 'ward').
    """
    unique_groups = data[group_column].unique()
    for group in unique_groups:
        # Filtrowanie danych dla aktualnej grupy
        subset = data[data[group_column] == group]
        
        # Obliczanie linkage dla aktualnej grupy
        linked = linkage(subset[x_cols], method=method)
        
        # Rysowanie dendrogramu
        plt.figure(figsize=(15, 8))
        dendrogram(linked,
                   orientation='top',
                   labels=subset[label_column].values,
                   distance_sort='descending',
                   show_leaf_counts=True)
        plt.title(f'Dendrogram for {group_column} = {group}')
        plt.xlabel('Sample index or (Cluster size)')
        plt.ylabel('Euclidean distances')
        plt.show()


# Funkcja do obliczania zadanej metryki
def calc_metric(metric_name, list, y_test, y_pred, **kwargs):
    var = metric_name(y_test, y_pred, **kwargs)
    list.append(var)
    return list


# Funkcja do wypisywania wartości 
def print_mean_calc_metric(metric_name, list, n_iterations):
    mean = np.mean(list)
    print(f"\nAverage {metric_name.__name__} after {n_iterations} iterations: {mean:.2f}")


# Funkcja do wypisywania macierzy pomyłek
def print_mean_confusion_matrix(list_of_cm, n_iterations):
    mean_cm = np.round(np.mean(list_of_cm, axis=0), decimals=2)
    print(f"\nAverage confusion matrix after {n_iterations} iterations:")
    print(mean_cm)