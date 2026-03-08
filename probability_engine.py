"""
╔══════════════════════════════════════════════════════════════════╗
║       probability_engine.py – LAYER 4: PROBABILITY ENGINE       ║
║      Kombiniert alle Signale zu einer finalen Wahrscheinlichkeit ║
╚══════════════════════════════════════════════════════════════════╝

LAYER 4 der Pipeline: Signal Fusion & Probability Calculation

Was macht dieses Modul?
→ Nimmt alle Signale aus Layer 1-3 und kombiniert sie zu EINEM Wert
→ "Wie wahrscheinlich ist es, dass dieser Trade profitabel ist?"
→ Gibt auch ein Konfidenzintervall zurück (Unsicherheitsbereich)

Methode: Bayesian Ensemble + Monte Carlo Simulation
→ Bayesian: Berücksichtigt Unsicherheit in jedem Signal
→ Monte Carlo: Simuliert 1000 mögliche Szenarien um Unsicherheit zu messen
"""

import numpy as np
import logging

log = logging.getLogger(__name__)


class ProbabilityEngine:
    """
    Kombiniert alle Handelssignale zu einer finalen Wahrscheinlichkeit.
    
    Stell dir vor: 3 Wetterexperten sagen das Wetter voraus.
    Der eine sagt 80% Regen, der andere 60%, der dritte 70%.
    Du kombinierst ihre Meinungen zu einer finalen Vorhersage.
    Das macht die ProbabilityEngine mit Handelssignalen.
    """

    def __init__(self, min_confidence: float = 0.65, mc_simulations: int = 1000):
        """
        Erstellt eine neue ProbabilityEngine.
        
        Parameter:
        - min_confidence:  Mindest-Wahrscheinlichkeit für Trade-Empfehlung
        - mc_simulations:  Anzahl Monte Carlo Simulationen (mehr = genauer)
        """
        self.min_confidence = min_confidence
        self.mc_simulations = mc_simulations

        # Gewichtung der Signale (aus config.py, hier als Default)
        # Orderflow ist am wichtigsten für kurzfristige Trades
        self.weights = {
            "regime":    0.35,
            "orderflow": 0.40,
            "noise":     0.25,
        }

    def calculate(self, regime: dict, orderflow: dict,
                  noise_level: float, price: float) -> dict:
        """
        Berechnet die finale Trade-Wahrscheinlichkeit.
        
        Parameter:
        - regime:      Ergebnis des RegimeClassifiers (dict mit label, confidence etc.)
        - orderflow:   Ergebnis des OrderflowAnalyzers (dict mit direction, strength etc.)
        - noise_level: Aktueller Noise-Level (0.0-1.0)
        - price:       Aktueller gefilterter Preis
        
        Rückgabe: Dictionary mit:
        - direction:   "long", "short" oder "neutral"
        - probability: Wahrscheinlichkeit 0.0-1.0
        - ci_low:      Untere Grenze des Konfidenzintervalls
        - ci_high:     Obere Grenze des Konfidenzintervalls
        - signal_quality: Gesamtqualität des Signals
        """
        # ── Schritt 1: Einzel-Signale extrahieren ───────────────
        
        # Regime-Signal: In welche Richtung und wie stark?
        regime_signal = self._regime_to_signal(regime)

        # Orderflow-Signal: Direkt aus dem Orderflow-Analyzer
        orderflow_signal = self._orderflow_to_signal(orderflow)

        # Noise-Signal: Weniger Noise = stärkeres Signal
        noise_signal = 1.0 - noise_level  # Invertiert: hoher Noise = niedriges Signal

        # ── Schritt 2: Richtung bestimmen ───────────────────────
        # Wenn Regime und Orderflow in die gleiche Richtung zeigen → stärkeres Signal
        direction = self._determine_direction(regime, orderflow)

        # ── Schritt 3: Gewichtete Kombination ───────────────────
        # Jedes Signal wird mit seiner Gewichtung multipliziert und addiert
        combined_probability = (
            self.weights["regime"]    * regime_signal    +
            self.weights["orderflow"] * orderflow_signal +
            self.weights["noise"]     * noise_signal
        )

        # ── Schritt 4: Monte Carlo Simulation ───────────────────
        # Simuliert viele mögliche Szenarien mit leicht variierenden Inputs
        # → Gibt uns ein Konfidenzintervall (Unsicherheitsbereich)
        mc_results = self._monte_carlo(
            base_probability=combined_probability,
            regime_confidence=regime["confidence"],
            orderflow_strength=orderflow["strength"],
        )

        return {
            "direction":      direction,
            "probability":    float(np.clip(combined_probability, 0, 1)),
            "ci_low":         float(mc_results["ci_low"]),
            "ci_high":        float(mc_results["ci_high"]),
            "signal_quality": float(mc_results["stability"]),
            "signals": {           # Detail-Aufschlüsselung für Debugging
                "regime":    regime_signal,
                "orderflow": orderflow_signal,
                "noise":     noise_signal,
            }
        }

    def _regime_to_signal(self, regime: dict) -> float:
        """
        Wandelt das Regime-Ergebnis in eine Signalstärke um (0.0-1.0).
        
        Nicht alle Regime sind gleich gut für Trades:
        - trending_up/down:  Starkes Signal (wir handeln mit dem Trend)
        - mean_reverting:    Mittleres Signal (Umkehr-Handel möglich)
        - high_volatility:   Schwaches Signal (zu gefährlich)
        """
        label = regime.get("label", "mean_reverting")
        confidence = regime.get("confidence", 0.5)

        # Basis-Signalstärke je nach Regime
        regime_strength = {
            "trending_up":     0.8,   # Starkes Signal – Trend folgen
            "trending_down":   0.8,   # Starkes Signal – Trend folgen
            "mean_reverting":  0.5,   # Mittleres Signal
            "high_volatility": 0.2,   # Schwaches Signal – gefährlich
        }

        base = regime_strength.get(label, 0.4)
        return base * confidence  # Mit Konfidenz gewichtet

    def _orderflow_to_signal(self, orderflow: dict) -> float:
        """
        Wandelt das Orderflow-Signal in eine Signalstärke um (0.0-1.0).
        """
        direction = orderflow.get("direction", "neutral")
        strength = orderflow.get("strength", 0.0)

        if direction == "neutral":
            return 0.3   # Neutrales Signal = 30% Basiswahrscheinlichkeit
        else:
            # long oder short = klares Signal, skaliert mit Stärke
            return 0.5 + strength * 0.5  # 0.5 bis 1.0

    def _determine_direction(self, regime: dict, orderflow: dict) -> str:
        """
        Bestimmt die finale Handelsrichtung basierend auf allen Signalen.
        
        Regel: Wenn Regime und Orderflow in die gleiche Richtung zeigen
               → Handeln wir in diese Richtung
               Sonst → Neutral (kein Trade)
        """
        regime_label = regime.get("label", "mean_reverting")
        orderflow_dir = orderflow.get("direction", "neutral")

        # Regime-Richtung bestimmen
        if regime_label == "trending_up":
            regime_dir = "long"
        elif regime_label == "trending_down":
            regime_dir = "short"
        else:
            regime_dir = "neutral"  # Kein klarer Trend

        # Wenn beide gleiche Richtung → starkes Signal
        if regime_dir == orderflow_dir and orderflow_dir != "neutral":
            return orderflow_dir

        # Wenn nur Orderflow ein Signal hat → schwächeres Signal
        if orderflow_dir != "neutral" and regime_dir == "neutral":
            return orderflow_dir

        # Wenn Widerspruch → neutral
        return "neutral"

    def _monte_carlo(self, base_probability: float,
                     regime_confidence: float,
                     orderflow_strength: float) -> dict:
        """
        Monte Carlo Simulation für Konfidenzintervall.
        
        Was ist Monte Carlo?
        → Stell dir vor, du würfelt 1000 Mal.
        → Jedes Mal veränderst du die Signale ein bisschen (zufällig).
        → Du schaust: In welchem Bereich liegen die Ergebnisse meistens?
        → Das ist dein Konfidenzintervall.
        
        Warum? → Um die Unsicherheit des Signals zu messen.
        Enge Intervalle = Starkes, verlässliches Signal
        Weite Intervalle = Unsicheres, schwaches Signal
        """
        n = self.mc_simulations

        # Unsicherheit jedes Signals (Standardabweichung für Simulation)
        # Höhere Konfidenz = kleinere Unsicherheit
        regime_uncertainty    = (1 - regime_confidence) * 0.2
        orderflow_uncertainty = (1 - orderflow_strength) * 0.15
        noise_uncertainty     = 0.1   # Noise-Signal hat immer etwas Unsicherheit

        # Simuliere n Szenarien mit zufälligen Abweichungen
        np.random.seed(None)  # Echter Zufall (kein fixer Seed)
        
        simulated_regime    = np.random.normal(base_probability, regime_uncertainty, n)
        simulated_orderflow = np.random.normal(base_probability, orderflow_uncertainty, n)
        simulated_noise     = np.random.normal(base_probability, noise_uncertainty, n)

        # Kombiniere die simulierten Werte (gleiche Gewichtung wie zuvor)
        simulated_combined = (
            self.weights["regime"]    * simulated_regime    +
            self.weights["orderflow"] * simulated_orderflow +
            self.weights["noise"]     * simulated_noise
        )

        # Auf gültigen Bereich begrenzen
        simulated_combined = np.clip(simulated_combined, 0, 1)

        # Konfidenzintervall: Bereich der mittleren 90% der Ergebnisse
        ci_low  = float(np.percentile(simulated_combined, 5))    # Untere 5%
        ci_high = float(np.percentile(simulated_combined, 95))   # Obere 95%

        # Stabilität: Wie eng ist das Intervall? (enger = stabiler)
        interval_width = ci_high - ci_low
        stability = float(np.clip(1.0 - interval_width * 2, 0, 1))

        return {
            "ci_low":    ci_low,
            "ci_high":   ci_high,
            "stability": stability,
            "mean":      float(np.mean(simulated_combined)),
            "std":       float(np.std(simulated_combined)),
        }
