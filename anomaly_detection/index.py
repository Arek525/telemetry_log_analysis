import pandas as pd
import numpy as np
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

def load_logs(file_path):
    """Wczytaj logi z pliku CSV"""
    df = pd.read_csv(file_path, sep=";")
    
    # Rozpakuj JSON z AttributesJson
    df["AttributesJson"] = df["AttributesJson"].apply(json.loads)
    attributes_df = pd.json_normalize(df["AttributesJson"])
    
    # Połącz z głównym DataFrame
    df = pd.concat(
        [df.drop(columns=["AttributesJson"]), attributes_df],
        axis=1
    )
    
    return df

def build_feature_vector(df):
    features = [
        'LatencyMs',
        'CpuUsage',
        'MemoryUsageMb',
        'DiskQueueLength',
        'NetworkErrors',
        'LocalQps',
        'RequestSizeBytes',
        'ResponseSizeBytes'
    ]
    
    # Sprawdź czy wszystkie kolumny istnieją
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        print(f"Brakujące kolumny: {missing_cols}")
    
    X = df[features].copy()
    
    # Obsłuż brakujące wartości
    X = X.fillna(X.mean())
    
    return X, features


def detect_anomalies_isolation_forest(X, y_true=None, contamination=None):
    """
    Detekcja anomalii za pomocą Isolation Forest z grid search
    
    Args:
        X: macierz cech
        y_true: rzeczywiste etykiety anomalii (jeśli są)
        contamination: proporcja anomalii (jeśli None, oblicza się z y_true)
    
    Returns:
        predictions: -1 dla anomalii, 1 dla normalnych
        scores: anomaly scores
        best_contamination: znaleziona optymalna wartość
    """
    # Normalizacja cech
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Jeśli nie podano contamination, spróbuj znaleźć optymalną wartość
    if contamination is None:
        if y_true is not None:
            actual_contamination = y_true.sum() / len(y_true)
            print(f"Rzeczywista proporcja anomalii: {actual_contamination:.6f}")
        else:
            actual_contamination = 0.1
        
        # Grid search - testuj szeroki zakres wartości (dostosowany do 13-200k anomalii)
        contam_values = [
            actual_contamination * 0.3,
            actual_contamination * 0.5,
            actual_contamination * 0.75,
            actual_contamination,
            actual_contamination * 1.5,
            actual_contamination * 2.0,
            actual_contamination * 3.0,
            actual_contamination * 5.0,
            actual_contamination * 10.0
        ]
        # Ogranicze do [0.00001, 0.5] - pozwala na bardzo małe proporcje
        contam_values = [max(0.00001, min(0.5, c)) for c in contam_values]
        
        best_f1 = -1
        best_contamination = contamination if contamination else 0.1
        best_predictions = None
        best_scores = None
        
        print("\nGrid Search - szukanie optymalnej wartości contamination:")
        print(f"{'Contamination':<15} {'Detected':<10} {'Precision':<12} {'Recall':<10} {'F1':<10}")
        print("-" * 57)
        
        for contam in sorted(set(contam_values)):
            # Więcej drzew = lepsza dokładność dla małych proporcji anomalii
            iso_forest = IsolationForest(contamination=contam, random_state=42, n_estimators=200, max_samples='auto')
            preds = iso_forest.fit_predict(X_scaled)
            scores = iso_forest.score_samples(X_scaled)
            
            y_pred = np.where(preds == -1, 1, 0)
            detected_count = y_pred.sum()
            
            if y_true is not None:
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                print(f"{contam:<15.6f} {detected_count:<10} {prec:<12.4f} {rec:<10.4f} {f1:<10.4f}")
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_contamination = contam
                    best_predictions = preds
                    best_scores = scores
            else:
                print(f"{contam:<15.6f} {detected_count:<10}")
                best_contamination = contam
                best_predictions = preds
                best_scores = scores
        
        print(f"\nNajlepsza wartość contamination: {best_contamination:.6f} (F1={best_f1:.4f})")
        predictions = best_predictions
        scores = best_scores
    else:
        iso_forest = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        predictions = iso_forest.fit_predict(X_scaled)
        scores = iso_forest.score_samples(X_scaled)
        best_contamination = contamination
    
    iso_forest_final = IsolationForest(contamination=best_contamination, random_state=42, n_estimators=200, max_samples='auto')
    iso_forest_final.fit(X_scaled)
    
    return predictions, scores, iso_forest_final, scaler, best_contamination


def detect_anomalies_zscore(X, threshold=3.5):
    """Detekcja anomalii za pomocą z-score - dostrojony threshold"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Wylicz z-scores
    z_scores = np.abs(X_scaled)
    
    # Punkt jest anomalią jeśli którekolwiek z jego z-scores > threshold
    predictions = np.where((z_scores > threshold).any(axis=1), -1, 1)
    
    # Średni z-score jako anomaly score
    scores = z_scores.mean(axis=1)
    scores = -scores  # Ujemne by były spójne z IF
    
    return predictions, scores, scaler


def detect_anomalies_percentile(X, y_true, percentile_range=[95, 97.5, 99, 99.5, 99.9]):
    """
    Detekcja anomalii bazująca na percentylach anomaly scores z IF
    Znajduje optymalny próg odcięcia
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Trenuj IF z auto contamination
    iso_forest = IsolationForest(contamination='auto', random_state=42, n_estimators=200, max_samples='auto')
    iso_forest.fit(X_scaled)
    scores = iso_forest.score_samples(X_scaled)
    
    best_f1 = -1
    best_threshold = None
    best_predictions = None
    
    print("\nPercentile Search - szukanie optymalnego progu:")
    print(f"{'Percentile':<12} {'Threshold':<12} {'Detected':<10} {'Precision':<12} {'Recall':<10} {'F1':<10}")
    print("-" * 68)
    
    for perc in percentile_range:
        threshold = np.percentile(scores, perc)
        preds = np.where(scores < threshold, -1, 1)
        
        y_pred = np.where(preds == -1, 1, 0)
        detected_count = y_pred.sum()
        
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        print(f"{perc:<12.1f} {threshold:<12.4f} {detected_count:<10} {prec:<12.4f} {rec:<10.4f} {f1:<10.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_predictions = preds
    
    print(f"\nNajlepszy percentyl: threshold={best_threshold:.4f}, F1={best_f1:.4f}")
    
    return best_predictions, scores, iso_forest, scaler





def evaluate_anomaly_detection(df, predictions, method_name="ISOLATION FOREST"):
    """
    Ocena wyników detekcji anomalii
    Porównanie z IsAnomaly oraz EventCode 998/999
    """
    
    # Przygotuj Ground Truth z IsAnomaly
    y_true_is_anomaly = df["IsAnomaly"].values
    
    # Przygotuj Ground Truth z EventCode (998/999 to anomalie)
    y_true_event_code = ((df["EventCode"] == 998) | (df["EventCode"] == 999)).astype(int).values
    
    # Zamień -1 (anomalia) na 1, 1 (normal) na 0 dla porównania
    y_pred = np.where(predictions == -1, 1, 0)
    
    print("\n" + "="*60)
    print(f"OCENA DETEKCJI ANOMALII - {method_name}")
    print("="*60)
    
    # Porównanie z IsAnomaly
    print("\n--- Porównanie z kolumną IsAnomaly ---")
    print(f"Liczba anomalii (IsAnomaly=1): {y_true_is_anomaly.sum()}")
    print(f"Liczba anomalii (Detector): {y_pred.sum()}")
    
    precision_is = precision_score(y_true_is_anomaly, y_pred, zero_division=0)
    recall_is = recall_score(y_true_is_anomaly, y_pred, zero_division=0)
    f1_is = f1_score(y_true_is_anomaly, y_pred, zero_division=0)
    
    print(f"\nPrecision:  {precision_is:.4f}")
    print(f"Recall:     {recall_is:.4f}")
    print(f"F1-Score:   {f1_is:.4f}")
    
    print("\nMacierz pomyłek (TN, FP, FN, TP):")
    cm = confusion_matrix(y_true_is_anomaly, y_pred)
    print(f"[[{cm[0,0]:<10} {cm[0,1]:<10}]")
    print(f" [{cm[1,0]:<10} {cm[1,1]:<10}]]")
    
    print("\nRaport klasyfikacji:")
    print(classification_report(y_true_is_anomaly, y_pred, target_names=['Normal', 'Anomaly'], zero_division=0))
    
    # Porównanie z EventCode 998/999
    print("\n--- Porównanie z EventCode 998/999 ---")
    print(f"Liczba anomalii (EventCode 998/999): {y_true_event_code.sum()}")
    
    if y_true_event_code.sum() > 0:
        precision_ec = precision_score(y_true_event_code, y_pred, zero_division=0)
        recall_ec = recall_score(y_true_event_code, y_pred, zero_division=0)
        f1_ec = f1_score(y_true_event_code, y_pred, zero_division=0)
        
        print(f"\nPrecision:  {precision_ec:.4f}")
        print(f"Recall:     {recall_ec:.4f}")
        print(f"F1-Score:   {f1_ec:.4f}")
    else:
        print("\nBrak anomalii w EventCode 998/999 w zbiorze danych")
        precision_ec = recall_ec = f1_ec = 0
    
    print("="*60 + "\n")
    
    return {
        'method': method_name,
        'is_anomaly': {
            'precision': precision_is,
            'recall': recall_is,
            'f1': f1_is
        },
        'event_code': {
            'precision': precision_ec,
            'recall': recall_ec,
            'f1': f1_ec
        }
    }


def add_anomaly_predictions_to_df(df, predictions, scores, method='IF'):
    """Dodaj predykcje anomalii do DataFrame"""
    df[f'{method}_Anomaly'] = np.where(predictions == -1, 1, 0)
    df[f'{method}_AnomalyScore'] = scores
    return df


def analyze_anomalies(df):
    """Analiza wykrytych anomalii"""
    # Znajdź kolumnę z anomaliami (IF_Anomaly, Z-Score_Anomaly, LOF_Anomaly)
    anomaly_col = None
    for col in df.columns:
        if '_Anomaly' in col and col != 'IsAnomaly':
            anomaly_col = col
            break
    
    if anomaly_col is None:
        anomaly_col = 'IF_Anomaly'
    
    anomalies = df[df[anomaly_col] == 1]
    
    print("\n--- Analiza wykrytych anomalii ---")
    print(f"\nLiczba anomalii: {len(anomalies)}")
    
    if len(anomalies) > 0:
        score_col = anomaly_col.replace('_Anomaly', '_AnomalyScore')
        
        print(f"\nTop anomalii (metodą {anomaly_col.replace('_Anomaly', '')}):")
        if score_col in df.columns:
            top_anomalies = anomalies.nsmallest(10, score_col)[
                ['Timestamp', 'EventCode', 'IsAnomaly', 'LatencyMs', 'CpuUsage', 
                 'MemoryUsageMb', 'DiskQueueLength', 'NetworkErrors', score_col]
            ]
        else:
            top_anomalies = anomalies.head(10)[
                ['Timestamp', 'EventCode', 'IsAnomaly', 'LatencyMs', 'CpuUsage', 
                 'MemoryUsageMb', 'DiskQueueLength', 'NetworkErrors']
            ]
        print(top_anomalies.to_string())
        
        print("\n\nRozkład EventCode w anomaliach:")
        print(anomalies['EventCode'].value_counts())
        
        print("\n\nRozkład scenariuszów w anomaliach:")
        print(anomalies['Scenario'].value_counts())


def anomaly_detection(df):
    print("\nBudowanie wektora cech...")
    X, features = build_feature_vector(df)
    print(f"Cechy: {features}")
    
    y_true = df["IsAnomaly"].values
    
    # ===== ISOLATION FOREST z Grid Search =====
    print("\n" + "="*60)
    print("METODA 1: ISOLATION FOREST (z Grid Search)")
    print("="*60)
    
    predictions_if, scores_if, iso_forest, scaler_if, best_contam = detect_anomalies_isolation_forest(
        X, y_true=y_true, contamination=None
    )
    
    df_if = add_anomaly_predictions_to_df(df.copy(), predictions_if, scores_if, method='IF')
    results_if = evaluate_anomaly_detection(df_if, predictions_if, "ISOLATION FOREST")
    
    # ===== Z-SCORE =====
    print("\n" + "="*60)
    print("METODA 2: Z-SCORE")
    print("="*60)
    
    predictions_zscore, scores_zscore, scaler_zscore = detect_anomalies_zscore(X, threshold=3)
    
    df_zscore = add_anomaly_predictions_to_df(df.copy(), predictions_zscore, scores_zscore, method='Z-Score')
    results_zscore = evaluate_anomaly_detection(df_zscore, predictions_zscore, "Z-SCORE")
    
    # ===== PERCENTILE-BASED IF =====
    print("\n" + "="*60)
    print("METODA 3: PERCENTILE-BASED ISOLATION FOREST")
    print("="*60)
    
    predictions_perc, scores_perc, iso_forest_perc, scaler_perc = detect_anomalies_percentile(
        X, y_true, percentile_range=[90, 92.5, 95, 97.5, 99, 99.5, 99.9]
    )
    
    df_perc = add_anomaly_predictions_to_df(df.copy(), predictions_perc, scores_perc, method='Percentile')
    results_perc = evaluate_anomaly_detection(df_perc, predictions_perc, "PERCENTILE-BASED IF")
    
    # ===== PORÓWNANIE METOD =====
    print("\n" + "="*70)
    print("PORÓWNANIE WSZYSTKICH METOD")
    print("="*70)
    
    comparison_data = {
        'Metoda': ['Isolation Forest', 'Z-Score', 'Percentile-IF'],
        'Precision': [results_if['is_anomaly']['precision'], results_zscore['is_anomaly']['precision'], results_perc['is_anomaly']['precision']],
        'Recall': [results_if['is_anomaly']['recall'], results_zscore['is_anomaly']['recall'], results_perc['is_anomaly']['recall']],
        'F1': [results_if['is_anomaly']['f1'], results_zscore['is_anomaly']['f1'], results_perc['is_anomaly']['f1']],
    }
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    # Wybierz najlepszą metodę
    best_method_idx = comparison_df['F1'].idxmax()
    best_method = comparison_df.loc[best_method_idx, 'Metoda']
    best_f1 = comparison_df.loc[best_method_idx, 'F1']
    
    print(f"\n🏆 Najlepsza metoda: {best_method} (F1={best_f1:.4f})")
    print("="*70 + "\n")
    
    # Użyj najlepszej metody do wyjścia
    if best_method == 'Isolation Forest':
        df_final = df_if
        predictions_final = predictions_if
    elif best_method == 'Z-Score':
        df_final = df_zscore
        predictions_final = predictions_zscore
    else:
        df_final = df_perc
        predictions_final = predictions_perc
    
    # Analiza anomalii
    analyze_anomalies(df_final)
    
    # Zapisz wyniki
    df_final.to_csv("anomaly_detection_results.csv", sep=";", index=False)
    print("\nWyniki zapisane do: anomaly_detection_results.csv")
    
    return df_final, iso_forest, scaler_if, {
        'isolation_forest': results_if,
        'zscore': results_zscore,
        'percentile': results_perc,
        'best_method': best_method
    }


def main():
    """Główna funkcja - do testowania standalone"""
    # Wczytaj dane (tylko jeśli uruchamiany samodzielnie)
    print("Wczytywanie danych...")
    df = load_logs("logs.csv")
    print(f"Wczytano {len(df)} rekordów")
    
    # Uruchom detekcję anomalii
    df_results, model, scaler, results = anomaly_detection(df)
    
    return df_results, results


if __name__ == "__main__":
    main()
