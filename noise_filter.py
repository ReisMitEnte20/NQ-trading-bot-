"""
╔══════════════════════════════════════════════════════════════════╗
║             noise_filter.py – LAYER 1: NOISE FILTER             ║
║         KDE / GMM zur Trennung von Signal und Rauschen          ║
╚══════════════════════════════════════════════════════════════════╝

LAYER 1 der Pipeline: Noise Probability Density Function

Was macht dieses Modul?
→ Tick-Daten enthalten viel "Rauschen" (zufällige Bewegungen ohne Information)
→ Dieser Filter trennt echte Preisbewegungen von Rauschen
→ Danach weiß der Bot: "Ist das gerade eine echte Bewegung oder nur Lärm?"

Methoden:
- KDE (Kernel Density Estimation): Schätzt die Wahrscheinlichkeitsdichte
- GMM (Gaussian Mixture Model): Modelliert Signal + Noise als separate Verteilungen
"""

import numpy as np
import logging
from scipy.stats import gaussian_kde              # Für KDE
from sklearn.mixture import GaussianMixture       # Für GMM

log = logging.getLogger(__name__)


class NoiseFilter:
    """
    Filtert Rauschen aus Preisdaten heraus.
    
    Stell dir vor: Ein Wasserfilter trennt sauberes Wasser von Schmutz.
    Der NoiseFilter trennt echte Preissignale von zufälligem Rauschen.
    """

    def __init__(self, method: str = "kde", bandwidth: float = 0.5, n_components: int = 3):
        """
        Erstellt einen neuen NoiseFilter.
        
        Parameter:
        - method:       "kde" oder "gmm"
        - bandwidth:    Stärke der KDE-Glättung (größer = mehr Glättung)
        - n_components: Anzahl GMM-Verteilungen (2 oder 3 empfohlen)
        """
        self.method = method
        self.bandwidth = bandwidth
        self.n_components = n_components

        self.kde_model = None        # Wird beim ersten filter()-Aufruf trainiert
        self.gmm_model = None        # Wird beim ersten filter()-Aufruf trainiert
        self._noise_level = 0.5      # Aktueller Noise-Level (0=kein Noise, 1=nur Noise)
        self._filtered_price = None  # Letzter gefilterter Preis

    def filter(self, price_series: np.ndarray) -> float:
        """
        Filtert die Preisserie und gibt den bereinigten aktuellen Preis zurück.
        
        Parameter:
        - price_series: Array mit den letzten N Preisen (z.B. letzte 100 Ticks)
        
        Rückgabe:
        - Gefilterter Preis (float) – "bereinigt" von Rauschen
        
        Beispiel:
            Preise:            [18000, 18001, 17999, 18005, 17998]
            Gefilterter Preis: 18000.6  (glatter, ohne Ausreißer)
        """
        if len(price_series) < 10:
            # Zu wenig Daten → ungefilterten letzten Preis zurückgeben
            return float(price_series[-1])

        if self.method == "kde":
            return self._filter_kde(price_series)
        elif self.method == "gmm":
            return self._filter_gmm(price_series)
        else:
            log.warning(f"Unbekannte Methode '{self.method}'. Nutze KDE.")
            return self._filter_kde(price_series)

    def _filter_kde(self, price_series: np.ndarray) -> float:
        """
        KDE-basierter Filter (Kernel Density Estimation).
        
        Wie funktioniert KDE?
        → Stell dir vor, du legst über jeden Datenpunkt eine kleine "Glocke" (Gauss-Kurve)
        → Alle Glocken zusammen ergeben eine glatte Wahrscheinlichkeitskurve
        → Der wahrscheinlichste Preis = der Hochpunkt der Kurve
        → Das ist unser "echter" Preis ohne Rauschen
        """
        # Preisveränderungen berechnen (Returns)
        # Wir analysieren Veränderungen, nicht absolute Preise
        returns = np.diff(price_series)

        try:
            # KDE auf die Preisveränderungen anpassen
            # bw_method kontrolliert wie "breit" jede Glocke ist
            kde = gaussian_kde(returns, bw_method=self.bandwidth)
            self.kde_model = kde

            # Noise-Level = Standardabweichung der Verteilung
            # Hohe Streuung = viel Noise
            std = np.std(returns)
            expected_std = np.abs(np.mean(returns)) * 10 + 1e-10

            # Noise-Level normalisiert auf 0-1
            # 0 = kein Noise (alle Bewegungen gleichförmig)
            # 1 = sehr viel Noise (chaotische Bewegungen)
            self._noise_level = min(1.0, std / (expected_std + std))

            # Gefilterten Preis berechnen:
            # Gleitender Durchschnitt der letzten Bars (gewichtet nach KDE-Dichte)
            weights = np.array([
                float(kde.evaluate([r])[0]) for r in returns[-20:]
            ])
            weights = weights / weights.sum()  # Normalisieren (sum = 1)

            # Letzter Preis + gewichteter Durchschnitt der letzten Returns
            filtered_return = np.dot(weights, returns[-20:])
            self._filtered_price = price_series[-1] + filtered_return * 0.5

        except Exception as e:
            log.warning(f"KDE-Fehler: {e}. Nutze rohen Preis.")
            self._filtered_price = float(price_series[-1])
            self._noise_level = 0.5

        return self._filtered_price

    def _filter_gmm(self, price_series: np.ndarray) -> float:
        """
        GMM-basierter Filter (Gaussian Mixture Model).
        
        Wie funktioniert GMM?
        → Nimmt an, dass Daten aus mehreren "Glocken" gemischt sind
        → Komponente 1 = echtes Signal (kleine, regelmäßige Bewegungen)
        → Komponente 2 = Noise (zufällige, unregelmäßige Ausreißer)
        → GMM trennt diese Komponenten voneinander
        
        Vorteil gegenüber KDE: Erkennt explizit Signal vs. Noise-Komponenten
        Nachteil: Braucht mehr Daten und ist etwas langsamer
        """
        returns = np.diff(price_series).reshape(-1, 1)  # GMM braucht 2D-Array

        try:
            # GMM trainieren
            self.gmm_model = GaussianMixture(
                n_components=self.n_components,
                covariance_type="full",   # Volle Kovarianzmatrix
                random_state=42,          # Für Reproduzierbarkeit
                max_iter=100,             # Maximale Optimierungsschritte
            )
            self.gmm_model.fit(returns)

            # Komponenten sortieren nach Varianz
            # Kleinste Varianz = Signal-Komponente
            # Größte Varianz  = Noise-Komponente
            variances = [
                self.gmm_model.covariances_[i][0, 0]
                for i in range(self.n_components)
            ]
            signal_idx = np.argmin(variances)   # Signal = kleinste Varianz
            noise_idx = np.argmax(variances)    # Noise  = größte Varianz

            # Noise-Level = Gewicht der Noise-Komponente
            # Wenn GMM 70% der Daten als Noise einordnet → noise_level = 0.7
            self._noise_level = float(self.gmm_model.weights_[noise_idx])

            # Signal-Mittelwert als gefilterter Return
            signal_mean = float(self.gmm_model.means_[signal_idx][0])
            self._filtered_price = price_series[-1] + signal_mean

        except Exception as e:
            log.warning(f"GMM-Fehler: {e}. Nutze KDE als Fallback.")
            return self._filter_kde(price_series)

        return self._filtered_price

    def get_noise_level(self) -> float:
        """
        Gibt den zuletzt berechneten Noise-Level zurück.
        
        Rückgabe: Float zwischen 0.0 und 1.0
        - 0.0 = absolut kein Rauschen (sehr selten)
        - 0.3 = wenig Rauschen → gute Handelsbedingungen
        - 0.6 = mittleres Rauschen → vorsichtig handeln
        - 0.8 = viel Rauschen → besser nicht handeln
        - 1.0 = nur Rauschen (z.B. Markt geschlossen)
        """
        return self._noise_level

    def get_signal_strength(self) -> float:
        """
        Kehrt den Noise-Level um → Signal-Stärke.
        
        Rückgabe: Float zwischen 0.0 und 1.0
        - 1.0 = starkes Signal, kein Noise
        - 0.0 = kein Signal, nur Noise
        """
        return 1.0 - self._noise_level
