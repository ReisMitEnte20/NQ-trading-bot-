"""
╔══════════════════════════════════════════════════════════════════╗
║          regime_classifier.py – LAYER 2: REGIME DETECTOR        ║
║         HMM erkennt den aktuellen Marktzustand                  ║
╚══════════════════════════════════════════════════════════════════╝

LAYER 2 der Pipeline: Market Regime Classifier

Was macht dieses Modul?
→ Märkte befinden sich immer in einem "Zustand" (Regime)
→ Der Bot muss wissen: Trend? Seitwärts? Hohe Volatilität?
→ Je nach Regime funktionieren verschiedene Strategien besser/schlechter

Die 4 Markt-Regime:
1. TRENDING_UP    → Markt steigt: Kaufsignale bevorzugen
2. TRENDING_DOWN  → Markt fällt: Verkaufssignale bevorzugen
3. MEAN_REVERTING → Seitwärts: Rückkehr zum Mittelwert handeln
4. HIGH_VOLATILITY → Chaos: Gar nicht handeln oder sehr kleine Positionen

Methode: Hidden Markov Model (HMM)
→ Der "verborgene" Zustand (Regime) wird aus sichtbaren Daten (Preise, ATR, RSI) geschätzt
"""

import numpy as np
import logging
import pickle
import os
from typing import Optional

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# HINWEIS ZU hmmlearn:
# hmmlearn lässt sich auf Python 3.14 noch nicht installieren (C++ Problem).
# Wir nutzen stattdessen ein eigenes Mini-HMM das in reinem Python geschrieben
# ist – kein C++ Compiler nötig, funktioniert auf allen Python-Versionen!
# ─────────────────────────────────────────────────────────────────────────────

HMM_AVAILABLE = True   # Wir nutzen unsere eigene Implementierung unten


class SimpleGaussianHMM:
    """
    Ein einfaches Hidden Markov Model in reinem Python.
    Ersetzt hmmlearn – kein C++ Compiler nötig!

    Wie funktioniert ein HMM?
    → Es gibt versteckte (hidden) Zustände (z.B. "Trending Up")
    → Wir sehen nur die Beobachtungen (z.B. ATR, RSI-Werte)
    → Das HMM lernt: Welche Beobachtungen kommen typischerweise
      in welchem Zustand vor?
    → Danach kann es aus neuen Beobachtungen den Zustand schätzen

    Trainingsalgorithmus: Baum-Welch (Expectation-Maximization)
    Vorhersage-Algorithmus: Viterbi
    """

    def __init__(self, n_components: int = 4, n_iter: int = 50, random_state: int = 42):
        self.n_components = n_components   # Anzahl versteckter Zustände
        self.n_iter = n_iter               # Trainings-Iterationen
        self.rng = np.random.default_rng(random_state)

        # Modell-Parameter (werden beim Training gelernt)
        self.means_       = None    # Mittlere Feature-Werte pro Zustand
        self.covars_      = None    # Varianz der Features pro Zustand
        self.transmat_    = None    # Übergangswahrscheinlichkeiten zwischen Zuständen
        self.startprob_   = None    # Startwahrscheinlichkeiten
        self._is_fitted   = False

    def fit(self, X: np.ndarray) -> "SimpleGaussianHMM":
        """
        Trainiert das HMM auf den Daten X.

        Parameter:
        - X: (N, D) Matrix – N Zeitpunkte, D Features

        Nutzt K-Means zur Initialisierung, dann Baum-Welch-Iteration.
        """
        N, D = X.shape
        K = self.n_components

        # ── Schritt 1: Initialisierung via K-Means ─────────────
        # K-Means gruppiert die Daten in K Cluster als Startpunkt
        labels = self._kmeans_init(X, K)

        # Initialisiere Mittelwerte aus K-Means-Clustern
        self.means_ = np.array([
            X[labels == k].mean(axis=0) if np.any(labels == k)
            else X[self.rng.integers(N)]
            for k in range(K)
        ])

        # Initialisiere Varianzen (Diagonale Kovarianz)
        self.covars_ = np.array([
            np.var(X[labels == k], axis=0) + 1e-6 if np.any(labels == k)
            else np.ones(D) * 0.1
            for k in range(K)
        ])

        # Gleichmäßige Übergangsmatrix als Start
        self.transmat_  = np.full((K, K), 1.0 / K)
        self.startprob_ = np.full(K, 1.0 / K)

        # ── Schritt 2: Baum-Welch (EM) Iteration ───────────────
        prev_log_prob = -np.inf
        for iteration in range(self.n_iter):
            # E-Schritt: Berechne Zustandswahrscheinlichkeiten
            gamma, xi, log_prob = self._e_step(X)

            # M-Schritt: Aktualisiere Modell-Parameter
            self._m_step(X, gamma, xi)

            # Konvergenz-Prüfung: Wenn Log-Likelihood sich kaum ändert → fertig
            if abs(log_prob - prev_log_prob) < 1e-4:
                break
            prev_log_prob = log_prob

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Sagt die wahrscheinlichsten Zustände für die Beobachtungen X vorher.
        Nutzt den Viterbi-Algorithmus.
        """
        if not self._is_fitted:
            raise RuntimeError("HMM muss erst trainiert werden (fit aufrufen)!")
        return self._viterbi(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Gibt die Wahrscheinlichkeit jedes Zustands für jede Beobachtung zurück.
        Rückgabe: (N, K) Matrix
        """
        gamma, _, _ = self._e_step(X)
        return gamma

    # ── Interne Hilfsmethoden ──────────────────────────────────

    def _gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
        """Berechnet die Gauss-Wahrscheinlichkeitsdichte für Vektor x."""
        D = len(x)
        diff = x - mean
        # Vermeidet numerische Probleme durch clipping
        var_safe = np.clip(var, 1e-6, None)
        log_p = -0.5 * (np.sum(diff**2 / var_safe) + np.sum(np.log(var_safe)) + D * np.log(2 * np.pi))
        return np.exp(np.clip(log_p, -500, 0))

    def _emission_probs(self, X: np.ndarray) -> np.ndarray:
        """Berechnet P(Beobachtung | Zustand) für alle Zeitpunkte."""
        N = len(X)
        K = self.n_components
        B = np.zeros((N, K))
        for t in range(N):
            for k in range(K):
                B[t, k] = self._gaussian_pdf(X[t], self.means_[k], self.covars_[k])
            # Normalisiere zur numerischen Stabilität
            row_sum = B[t].sum()
            if row_sum > 0:
                B[t] /= row_sum
            else:
                B[t] = 1.0 / K
        return B

    def _forward_backward(self, B: np.ndarray):
        """Forward-Backward Algorithmus für den E-Schritt."""
        N, K = B.shape
        A = self.transmat_
        pi = self.startprob_

        # Forward-Pass: alpha[t,k] = P(o_1..o_t, q_t=k)
        alpha = np.zeros((N, K))
        alpha[0] = pi * B[0]
        scale = np.zeros(N)
        scale[0] = alpha[0].sum() or 1e-300
        alpha[0] /= scale[0]

        for t in range(1, N):
            alpha[t] = (alpha[t-1] @ A) * B[t]
            scale[t] = alpha[t].sum() or 1e-300
            alpha[t] /= scale[t]

        # Backward-Pass: beta[t,k] = P(o_{t+1}..o_T | q_t=k)
        beta = np.zeros((N, K))
        beta[-1] = 1.0
        for t in range(N-2, -1, -1):
            beta[t] = A @ (B[t+1] * beta[t+1])
            b_sum = beta[t].sum() or 1e-300
            beta[t] /= b_sum

        log_prob = np.sum(np.log(scale + 1e-300))
        return alpha, beta, log_prob

    def _e_step(self, X: np.ndarray):
        """E-Schritt: Berechnet Zustandswahrscheinlichkeiten."""
        B = self._emission_probs(X)
        alpha, beta, log_prob = self._forward_backward(B)
        N, K = B.shape
        A = self.transmat_

        # gamma[t,k] = P(q_t=k | Beobachtungen)
        gamma = alpha * beta
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma /= np.where(gamma_sum > 0, gamma_sum, 1)

        # xi[t,i,j] = P(q_t=i, q_{t+1}=j | Beobachtungen)
        xi = np.zeros((N-1, K, K))
        for t in range(N-1):
            xi[t] = (alpha[t:t+1].T * A) * (B[t+1] * beta[t+1])
            xi_sum = xi[t].sum()
            if xi_sum > 0:
                xi[t] /= xi_sum

        return gamma, xi, log_prob

    def _m_step(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray):
        """M-Schritt: Aktualisiert Modell-Parameter."""
        K = self.n_components
        N, D = X.shape

        # Startwahrscheinlichkeiten
        self.startprob_ = gamma[0] + 1e-6
        self.startprob_ /= self.startprob_.sum()

        # Übergangsmatrix
        xi_sum = xi.sum(axis=0)
        self.transmat_ = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + 1e-6)

        # Mittelwerte und Varianzen
        for k in range(K):
            w = gamma[:, k:k+1]        # Gewichte für Zustand k
            w_sum = w.sum() + 1e-6
            self.means_[k]  = (w * X).sum(axis=0) / w_sum
            diff = X - self.means_[k]
            self.covars_[k] = (w * diff**2).sum(axis=0) / w_sum + 1e-6

    def _viterbi(self, X: np.ndarray) -> np.ndarray:
        """Viterbi-Algorithmus: Findet den wahrscheinlichsten Zustandspfad."""
        N = len(X)
        K = self.n_components
        B = self._emission_probs(X)
        A = np.log(self.transmat_ + 1e-300)
        pi = np.log(self.startprob_ + 1e-300)

        # Log-Wahrscheinlichkeiten für numerische Stabilität
        log_B = np.log(B + 1e-300)

        delta = np.zeros((N, K))
        psi   = np.zeros((N, K), dtype=int)

        delta[0] = pi + log_B[0]
        for t in range(1, N):
            for k in range(K):
                scores = delta[t-1] + A[:, k]
                psi[t, k]   = np.argmax(scores)
                delta[t, k] = scores[psi[t, k]] + log_B[t, k]

        # Backtracking: Bester Pfad rückwärts verfolgen
        path = np.zeros(N, dtype=int)
        path[-1] = np.argmax(delta[-1])
        for t in range(N-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]

        return path

    def _kmeans_init(self, X: np.ndarray, K: int, n_iter: int = 20) -> np.ndarray:
        """Einfaches K-Means zur Initialisierung der HMM-Zustände."""
        N = X.shape[0]
        # Zufällige Startpunkte
        idx = self.rng.choice(N, K, replace=False)
        centers = X[idx].copy()

        labels = np.zeros(N, dtype=int)
        for _ in range(n_iter):
            # Jedem Punkt den nächsten Cluster zuweisen
            dists = np.array([np.linalg.norm(X - c, axis=1) for c in centers])
            labels = np.argmin(dists, axis=0)
            # Cluster-Zentren neu berechnen
            for k in range(K):
                if np.any(labels == k):
                    centers[k] = X[labels == k].mean(axis=0)
        return labels


class RegimeClassifier:
    """
    Erkennt den aktuellen Marktzustand (Regime) mithilfe eines HMM.
    
    Stell dir vor: Ein Arzt schaut auf Fieberkurven und erkennt:
    "Der Patient ist in Zustand X (z.B. Fieber steigend)"
    Genauso schaut der RegimeClassifier auf Marktdaten und erkennt
    den aktuellen Markt-Zustand.
    """

    # Namen der Regime (werden den HMM-Zuständen zugeordnet)
    REGIME_NAMES = {
        0: "trending_up",
        1: "trending_down",
        2: "mean_reverting",
        3: "high_volatility",
    }

    def __init__(self, n_regimes: int = 4, model_path: Optional[str] = None):
        """
        Erstellt einen neuen RegimeClassifier.
        
        Parameter:
        - n_regimes:   Anzahl verschiedener Markt-Zustände (4 empfohlen)
        - model_path:  Pfad zu einem vortrainierten Modell (None = neu trainieren)
        """
        self.n_regimes = n_regimes
        self.model_path = model_path
        self.hmm_model = None
        self.is_trained = False
        self.history = []  # Speichert vergangene Features für HMM-Training

        # Versuche gespeichertes Modell zu laden
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
            log.info(f"✓ Regime-Modell geladen aus: {model_path}")
        else:
            log.info("Kein vortrainiertes Regime-Modell. Wird online trainiert.")

    def predict(self, prices: np.ndarray | float, features: dict) -> dict:
        """
        Bestimmt das aktuelle Markt-Regime.
        
        Parameter:
        - prices:   Array mit Preishistorie (oder einzelner Preis)
        - features: Dictionary mit Indikatoren (ATR, RSI, ADX, volume_ratio)
        
        Rückgabe: Dictionary mit:
        - label:      Name des Regimes ("trending_up", "high_volatility", etc.)
        - regime_id:  Numerische ID des Regimes (0-3)
        - confidence: Wie sicher ist die Vorhersage (0.0 - 1.0)
        - probabilities: Wahrscheinlichkeit jedes Regimes
        """
        # Feature-Vektor aus Indikatoren bauen
        feature_vector = self._build_features(features)
        self.history.append(feature_vector)

        # Warte bis genug Daten für Training vorhanden sind
        if len(self.history) < 50:
            return self._rule_based_regime(features)

        # HMM trainieren wenn noch nicht trainiert (nach 50 Datenpunkten)
        if not self.is_trained and HMM_AVAILABLE:
            self._train_hmm()

        # Regime vorhersagen
        if self.is_trained and HMM_AVAILABLE:
            return self._hmm_predict(feature_vector)
        else:
            # Fallback: Regelbasierte Regime-Erkennung
            return self._rule_based_regime(features)

    def _build_features(self, features: dict) -> np.ndarray:
        """
        Baut einen Feature-Vektor aus dem Indikatoren-Dictionary.
        
        Der HMM braucht die Daten als Array in immer gleicher Reihenfolge.
        Hier normalisieren wir die Werte auch (alle auf ähnliche Skala).
        """
        atr = features.get("atr", 10.0)
        rsi = features.get("rsi", 50.0)
        adx = features.get("adx", 20.0)
        volume_ratio = features.get("volume_ratio", 1.0)

        # Normalisierung:
        # ATR auf typischen Bereich skalieren (NQ ATR meist 5-50)
        atr_normalized = np.clip(atr / 20.0, 0, 5)
        # RSI ist bereits 0-100, auf 0-1 bringen
        rsi_normalized = rsi / 100.0
        # ADX ist 0-100, auf 0-1
        adx_normalized = np.clip(adx / 50.0, 0, 2)
        # Volume Ratio meist 0.5-3.0
        vol_normalized = np.clip(volume_ratio / 2.0, 0, 3)

        return np.array([atr_normalized, rsi_normalized, adx_normalized, vol_normalized])

    def _train_hmm(self):
        """
        Trainiert das Hidden Markov Model auf den gesammelten Daten.
        Nutzt unsere eigene SimpleGaussianHMM – kein hmmlearn nötig!
        """
        try:
            log.info("Trainiere HMM Regime-Classifier (eigene Implementierung)...")
            X = np.array(self.history)

            self.hmm_model = SimpleGaussianHMM(
                n_components=self.n_regimes,
                n_iter=50,
                random_state=42,
            )
            self.hmm_model.fit(X)
            self.is_trained = True
            log.info(f"✓ HMM trainiert mit {len(self.history)} Datenpunkten")

            if self.model_path:
                self._save_model(self.model_path)

        except Exception as e:
            log.error(f"HMM Training fehlgeschlagen: {e}")
            log.info("Nutze regelbasierten Fallback.")
            self.is_trained = False

    def _hmm_predict(self, feature_vector: np.ndarray) -> dict:
        """
        Sagt das aktuelle Regime mit dem trainierten HMM vorher.
        """
        try:
            # Wir brauchen die letzten N Punkte für Viterbi
            X = np.array(self.history[-50:])

            # Zustandssequenz per Viterbi
            state_sequence = self.hmm_model.predict(X)
            regime_id = int(state_sequence[-1])

            # Wahrscheinlichkeiten per Forward-Backward
            posteriors = self.hmm_model.predict_proba(X)[-1]

            regime_label = self._map_regime_to_name(regime_id, feature_vector)

            return {
                "label":         regime_label,
                "regime_id":     regime_id,
                "confidence":    float(posteriors[regime_id]),
                "probabilities": {
                    self.REGIME_NAMES.get(i, f"regime_{i}"): float(p)
                    for i, p in enumerate(posteriors)
                },
            }

        except Exception as e:
            log.warning(f"HMM Prediction fehlgeschlagen: {e}")
            return self._rule_based_regime({"atr": 10, "rsi": 50, "adx": 20})

    def _map_regime_to_name(self, regime_id: int, features: np.ndarray) -> str:
        """
        Mappt eine HMM-Zustandsnummer auf einen sinnvollen Namen.
        
        Problem: HMM nummeriert Zustände willkürlich (0, 1, 2, 3).
        Wir müssen herausfinden welche Nummer zu "Trending Up" etc. passt.
        
        Lösung: Wir schauen uns die Merkmale des Zustands an und ordnen zu.
        """
        # Mittelwerte der Features für diesen Zustand aus dem Modell holen
        if self.hmm_model and hasattr(self.hmm_model, "means_"):
            state_mean = self.hmm_model.means_[regime_id]
            # state_mean = [atr_norm, rsi_norm, adx_norm, vol_norm]
            atr_norm = state_mean[0]
            rsi_norm = state_mean[1]
            adx_norm = state_mean[2]

            # Mapping basierend auf typischen Indikatorwerten
            if atr_norm > 1.5:                          # Sehr hohe Volatilität
                return "high_volatility"
            elif adx_norm > 0.8 and rsi_norm > 0.6:    # Starker Trend + RSI hoch
                return "trending_up"
            elif adx_norm > 0.8 and rsi_norm < 0.4:    # Starker Trend + RSI niedrig
                return "trending_down"
            else:                                        # Kein klarer Trend
                return "mean_reverting"

        return self.REGIME_NAMES.get(regime_id, "unknown")

    def _rule_based_regime(self, features: dict) -> dict:
        """
        Regelbasierte Regime-Erkennung (Fallback wenn HMM nicht verfügbar).
        
        Einfache Regeln basierend auf klassischen Indikatoren.
        Nicht so präzise wie HMM, aber funktioniert immer.
        
        Regeln:
        - ATR sehr hoch                     → High Volatility
        - ADX > 30 und RSI > 60             → Trending Up
        - ADX > 30 und RSI < 40             → Trending Down
        - Alles andere                      → Mean Reverting
        """
        atr = features.get("atr", 10.0)
        rsi = features.get("rsi", 50.0)
        adx = features.get("adx", 20.0)

        # Regime bestimmen
        if atr > 30:                          # Sehr hohe Volatilität für NQ
            label = "high_volatility"
            confidence = min(1.0, atr / 50.0)
        elif adx > 30 and rsi > 60:           # Aufwärtstrend
            label = "trending_up"
            confidence = min(1.0, (adx - 25) / 25 * 0.7 + (rsi - 50) / 50 * 0.3)
        elif adx > 30 and rsi < 40:           # Abwärtstrend
            label = "trending_down"
            confidence = min(1.0, (adx - 25) / 25 * 0.7 + (50 - rsi) / 50 * 0.3)
        else:                                 # Seitwärtsmarkt
            label = "mean_reverting"
            confidence = max(0.3, 1.0 - adx / 30.0)

        confidence = float(np.clip(confidence, 0.1, 1.0))

        # Wahrscheinlichkeiten simulieren
        probs = {"trending_up": 0.1, "trending_down": 0.1,
                 "mean_reverting": 0.1, "high_volatility": 0.1}
        probs[label] = confidence
        # Rest gleichmäßig auf andere verteilen
        remaining = (1.0 - confidence) / 3.0
        for key in probs:
            if key != label:
                probs[key] = remaining

        return {
            "label":         label,
            "regime_id":     list(self.REGIME_NAMES.values()).index(label),
            "confidence":    confidence,
            "probabilities": probs,
        }

    def _save_model(self, path: str):
        """Speichert das trainierte HMM-Modell als Datei."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.hmm_model, f)
        log.info(f"HMM-Modell gespeichert: {path}")

    def _load_model(self, path: str):
        """Lädt ein gespeichertes HMM-Modell."""
        with open(path, "rb") as f:
            self.hmm_model = pickle.load(f)
        self.is_trained = True
