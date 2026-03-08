"""
╔══════════════════════════════════════════════════════════════════╗
║        orderflow_analyzer.py – LAYER 3: ORDERFLOW               ║
║         Orderbuch-Analyse und Muster-Erkennung                  ║
╚══════════════════════════════════════════════════════════════════╝

LAYER 3 der Pipeline: Orderflow Heatmap Analyzer

Was macht dieses Modul?
→ Analysiert das Orderbuch (Level-2 Daten / DOM)
→ Erkennt Muster wie:
   • Absorption:    Großer Käufer schluckt alle Verkaufsorders (bullisch)
   • Spoofing:      Große gefälschte Orders die gleich storniert werden
   • Iceberg-Order: Versteckte große Order (nur kleiner Teil sichtbar)
   • Imbalance:     Viel mehr Käufer als Verkäufer (oder umgekehrt)

Im echten Leben würde hier ein CNN (Convolutional Neural Net) das
Orderbuch-Bild analysieren. Wir nutzen regelbasierte Analyse als Start.
"""

import numpy as np
import logging
from typing import Optional

log = logging.getLogger(__name__)


class OrderflowAnalyzer:
    """
    Analysiert das Orderbuch auf Handelsmuster.
    
    Stell dir vor: Du schaust in ein Restaurant-Reservierungsbuch.
    Du kannst sehen ob viele Leute kommen (viele Reservierungen = bullisch)
    oder ob Tische storniert werden (bearisch).
    Das Orderbuch ist das Reservierungsbuch des Marktes.
    """

    def __init__(self, depth: int = 10, model_path: Optional[str] = None):
        """
        Erstellt einen neuen OrderflowAnalyzer.
        
        Parameter:
        - depth:      Wie viele Orderbuch-Level analysiert werden (5-10)
        - model_path: Pfad zu einem vortrainierten CNN (None = regelbasiert)
        """
        self.depth = depth
        self.model_path = model_path
        self.cnn_model = None
        self.orderbook_history = []  # Speichert letzte Orderbücher für Muster-Analyse

        if model_path:
            log.warning("CNN-Modell-Laden noch nicht implementiert. Nutze regelbasierte Analyse.")
        
        log.info(f"OrderflowAnalyzer bereit (Tiefe: {depth} Level)")

    def analyze(self, orderbook: dict, recent_trades: list) -> dict:
        """
        Analysiert das aktuelle Orderbuch und gibt ein Signal zurück.
        
        Parameter:
        - orderbook:     Dictionary mit "bids" und "asks" (jeweils Liste von price/size)
        - recent_trades: Liste der letzten ausgeführten Trades
        
        Rückgabe: Dictionary mit:
        - direction: "long", "short" oder "neutral"
        - strength:  Signalstärke 0.0-1.0
        - pattern:   Erkanntes Muster (z.B. "absorption", "imbalance")
        - details:   Weitere Details für Debugging
        """
        # Orderbuch-Snapshot speichern (für Heatmap-Analyse über Zeit)
        self.orderbook_history.append(orderbook)
        if len(self.orderbook_history) > 50:
            self.orderbook_history.pop(0)  # Alte Snapshots löschen

        # Basis-Metriken berechnen
        metrics = self._calculate_metrics(orderbook)

        # Muster erkennen
        pattern, pattern_strength = self._detect_pattern(metrics, recent_trades)

        # Richtung und Stärke bestimmen
        direction, strength = self._determine_signal(metrics, pattern, pattern_strength)

        result = {
            "direction": direction,
            "strength":  strength,
            "pattern":   pattern,
            "details": {
                "bid_volume":    metrics["total_bid_volume"],
                "ask_volume":    metrics["total_ask_volume"],
                "imbalance":     metrics["imbalance"],
                "bid_ask_ratio": metrics["bid_ask_ratio"],
                "spread":        metrics["spread"],
            }
        }

        return result

    def _calculate_metrics(self, orderbook: dict) -> dict:
        """
        Berechnet grundlegende Orderbuch-Metriken.
        
        Diese Zahlen helfen uns zu verstehen:
        - Wie viel will wer kaufen/verkaufen?
        - Gibt es eine Ungleichgewicht (Imbalance)?
        - Wie weit ist der Spread?
        """
        bids = orderbook.get("bids", [])  # Kauforders (Preis, Menge)
        asks = orderbook.get("asks", [])  # Verkaufsorders (Preis, Menge)

        if not bids or not asks:
            return self._empty_metrics()

        # Gesamtvolumen auf Bid- und Ask-Seite
        total_bid = sum(b["size"] for b in bids)
        total_ask = sum(a["size"] for a in asks)

        # Bid/Ask-Verhältnis
        # > 1.5 = mehr Käufer als Verkäufer (bullisch)
        # < 0.67 = mehr Verkäufer als Käufer (bearisch)
        ratio = total_bid / (total_ask + 1e-10)

        # Imbalance: Wie ungleich verteilt ist das Volumen?
        # +1.0 = nur Käufer, -1.0 = nur Verkäufer, 0.0 = ausgeglichen
        imbalance = (total_bid - total_ask) / (total_bid + total_ask + 1e-10)

        # Spread (Differenz zwischen bestem Ask und bestem Bid)
        best_bid = bids[0]["price"] if bids else 0
        best_ask = asks[0]["price"] if asks else 0
        spread = best_ask - best_bid

        # Volumen-Konzentration: Wie viel Volumen liegt nahe am Marktpreis?
        # Große Orders nahe dem Preis können den Markt "absorbieren"
        near_bid_vol = sum(b["size"] for b in bids[:3])  # Nur die 3 besten Levels
        near_ask_vol = sum(a["size"] for a in asks[:3])

        return {
            "total_bid_volume": total_bid,
            "total_ask_volume": total_ask,
            "bid_ask_ratio":    ratio,
            "imbalance":        imbalance,
            "spread":           spread,
            "near_bid_volume":  near_bid_vol,
            "near_ask_volume":  near_ask_vol,
            "best_bid":         best_bid,
            "best_ask":         best_ask,
        }

    def _detect_pattern(self, metrics: dict, recent_trades: list) -> tuple[str, float]:
        """
        Erkennt spezifische Orderbuch-Muster.
        
        Muster und ihre Bedeutung:
        
        1. ABSORPTION (bullisch/bearisch):
           → Große Sellers im Orderbuch, aber Preis bewegt sich nicht runter
           → Bedeutet: Jemand kauft alles auf (Absorption)
           → Signal: Kurz danach oft schnelle Aufwärtsbewegung
        
        2. IMBALANCE (bullisch/bearisch):
           → Deutlich mehr Volumen auf einer Seite
           → Signal: Markt wird wahrscheinlich in Richtung der größeren Seite gehen
        
        3. SPOOFING (neutralisierend):
           → Sehr große Orders tauchen auf und verschwinden schnell wieder
           → Bedeutet: Manipulation, Vorsicht!
        
        4. NORMAL:
           → Kein besonderes Muster erkennbar
        """
        imbalance = metrics.get("imbalance", 0)
        ratio = metrics.get("bid_ask_ratio", 1.0)

        # IMBALANCE erkennen
        if abs(imbalance) > 0.4:
            strength = min(1.0, abs(imbalance) * 2)
            return "imbalance", strength

        # ABSORPTION erkennen (vereinfacht)
        # Echte Absorption: Preis bewegt sich nicht trotz großem Volumen auf einer Seite
        if len(self.orderbook_history) >= 5:
            # Vergleiche aktuelles und vergangenes Volumen
            old_ask = sum(a["size"] for a in self.orderbook_history[-5].get("asks", []))
            new_ask = metrics["total_ask_volume"]
            
            if old_ask > 0 and new_ask < old_ask * 0.6:
                # Ask-Volumen stark gesunken → Absorption durch Käufer!
                return "absorption_bullish", 0.7

        # KEIN besonderes Muster
        return "normal", 0.3

    def _determine_signal(self, metrics: dict, pattern: str, pattern_strength: float) -> tuple[str, float]:
        """
        Übersetzt Metriken und Muster in ein konkretes Handelssignal.
        
        Rückgabe:
        - direction: "long", "short" oder "neutral"
        - strength:  0.0 bis 1.0 (wie stark ist das Signal?)
        """
        imbalance = metrics.get("imbalance", 0)

        # Muster-basierte Signale
        if pattern == "absorption_bullish":
            return "long", pattern_strength

        elif pattern == "imbalance":
            if imbalance > 0:   # Mehr Käufer
                return "long", pattern_strength
            else:               # Mehr Verkäufer
                return "short", pattern_strength

        elif pattern == "spoofing":
            return "neutral", 0.1   # Vorsicht bei Spoofing!

        # Basis-Signal aus Imbalance wenn kein Muster
        if imbalance > 0.2:
            return "long", min(0.6, abs(imbalance))
        elif imbalance < -0.2:
            return "short", min(0.6, abs(imbalance))

        return "neutral", 0.1

    def _empty_metrics(self) -> dict:
        """Gibt leere Metriken zurück wenn kein Orderbuch verfügbar."""
        return {
            "total_bid_volume": 0, "total_ask_volume": 0,
            "bid_ask_ratio": 1.0,  "imbalance": 0.0,
            "spread": 0.0,         "near_bid_volume": 0,
            "near_ask_volume": 0,  "best_bid": 0,  "best_ask": 0,
        }
