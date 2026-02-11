# Analiza Trendu (LatencyMs - OrderService)
Na podstawie wygenerowanego wykresu `trend_analysis.png`:

### Co przedstawia wykres:
1.  **Niebieska linia (LatencyMs)**: To surowe dane (mediana z 2-minutowych okien). Widać na niej "szpilki" (nagłe skoki opóźnień) oraz dłuższe okresy podwyższonych wartości.
2.  **Pomarańczowa przerywana linia (Smoothed)**: To wygładzony przebieg metryki (średnia/mediana krocząca), który służy algorytmowi do ignorowania chwilowych szumów.
3.  **Czerwone kropki (Detected Trend)**: To momenty, w których nasz algorytm `detect_trend` uznał, że występuje **rosnący trend**.
    *   Widać skupiska czerwonych kropek głównie na początku wykresu oraz w kilku późniejszych miejscach.
    *   Oznacza to, że w tych okresach opóźnienie systematycznie rosło przez co najmniej 5 kolejnych okien czasowych (`min_run=5`).
4.  **Zielone krzyżyki (Ground Truth)**: To miejsca, gdzie w danych syntetycznych flaga `TrendWindow` była ustawiona na `True` (czyli tam, gdzie generator logów "wiedział", że generuje trend).

### Analiza skuteczności:
*   **Trafienia (True Positives)**: Czerwone kropki pokrywają się z zielonymi krzyżykami (głównie na początku i w środku). Algorytm poprawnie wykrył te trendy.
*   **Brak detekcji (False Negatives)**: Są miejsca z zielonymi krzyżykami (np. te bardzo wysokie pojedyncze piki), gdzie nie ma czerwonych kropek. Algorytm ich nie wykrył, prawdopodobnie dlatego, że wzrost był zbyt gwałtowny (skokowy), a nie "trendowy" (stopniowy), lub trwał zbyt krótko (mniej niż 5 okien).
*   **Nadmiarowe detekcje (False Positives)**: Jeśli widzisz czerwone kropki tam, gdzie nie ma zielonych, algorytm "przewrażliwił się" na zwykłe wahania. Na tym wykresie precyzja (`Precision: 1.000`) sugeruje, że każda czerwona kropka trafiła w zielony obszar, więc fałszywych alarmów prawie nie ma.

### Wniosek
Algorytm dobrze wykrywa stabilne, narastające problemy (trendy), ale ignoruje nagłe, jednorazowe skoki (co jest zazwyczaj pożądanym zachowaniem w analizie trendów).
