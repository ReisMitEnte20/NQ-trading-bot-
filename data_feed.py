"""
╔══════════════════════════════════════════════════════════════════╗
║               data_feed.py – MARKTDATEN MODUL                   ║
║    Holt Tick-Daten, Preise und Orderbuch vom Broker             ║
╚══════════════════════════════════════════════════════════════════╝

Dieses Modul ist der "Eingang" des Bots.
Es holt alle nötigen Marktdaten und bereitet sie auf.

Im BACKTEST-Modus: Liest historische Daten aus einer CSV-Datei
Im LIVE-Modus:     Verbindet sich mit dem Broker (Rithmic/CQG/IB)
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime

log = logging.getLogger(__name__)


class DataFeed:
    """
    Der DataFeed ist für alle Marktdaten zuständig.
    
    Er liefert bei jedem Aufruf von get_latest_data() ein Dictionary
    mit allen Informationen die der Bot braucht:
    - Aktueller Preis
    - Bid/Ask Spread
    - Volumen
    - Orderbuch (Level 2 Daten)
    - Letzte ausgeführte Trades
    - Indikatoren (ATR, ADX, RSI etc.)
    """

    def __init__(self, symbol: str, mode: str, data_path: str = None):
        """
        Initialisiert den DataFeed.
        
        Parameter:
        - symbol:    Das Handelsinstrument (z.B. "NQ")
        - mode:      "backtest" oder "live"
        - data_path: Pfad zur CSV-Datei (nur für Backtest)
        """
        self.symbol = symbol
        self.mode = mode
        self.data_path = data_path
        self.current_index = 0      # Zähler für Backtest (wo in der CSV sind wir?)
        self.data = None            # Hier werden die geladenen Daten gespeichert

        if mode == "backtest":
            self._load_backtest_data()
        elif mode == "live":
            self._connect_live()
        else:
            raise ValueError(f"Unbekannter Modus: {mode}. Nutze 'backtest' oder 'live'.")

    def _load_backtest_data(self):
        """
        Lädt historische Daten aus einer CSV-Datei.
        
        Die CSV-Datei sollte folgende Spalten haben:
        timestamp, open, high, low, close, volume
        
        Tipp: Historische NQ-Daten bekommst du z.B. von:
        - Rithmic (wenn du Account hast)
        - NinjaTrader (Export-Funktion)
        - TradeStation
        - Oder einfach mit Yahoo Finance (yfinance Python-Paket)
        """
        try:
            log.info(f"Lade Backtest-Daten aus: {self.data_path}")
            self.data = pd.read_csv(self.data_path, parse_dates=["timestamp"])
            self.data = self.data.sort_values("timestamp").reset_index(drop=True)

            # Technische Indikatoren vorberechnen
            self.data = self._add_indicators(self.data)

            log.info(f"✓ {len(self.data)} Datenpunkte geladen "
                     f"({self.data['timestamp'].iloc[0]} bis "
                     f"{self.data['timestamp'].iloc[-1]})")

        except FileNotFoundError:
            # Wenn keine echte CSV vorhanden → Demo-Daten generieren
            log.warning(f"Datei '{self.data_path}' nicht gefunden!")
            log.warning("Generiere synthetische Demo-Daten für Tests...")
            self.data = self._generate_demo_data()
            self.data = self._add_indicators(self.data)
            log.info(f"✓ {len(self.data)} Demo-Datenpunkte generiert")

    def _generate_demo_data(self, n_bars: int = 2000) -> pd.DataFrame:
        """
        Generiert synthetische NQ-Preisdaten für Tests.
        Diese Funktion brauchst du NUR wenn du noch keine echten Daten hast.
        
        Die Demo-Daten simulieren realistische NQ-Preisbewegungen mit:
        - Zufälligem Preispfad (Random Walk)
        - Realistischer Volatilität (~30 Punkte/Tag für NQ)
        - Volumen-Simulation
        """
        np.random.seed(42)  # Seed = immer gleiche Demo-Daten (für Reproduzierbarkeit)

        # Startpreis für NQ (ungefähr aktuelles Level)
        start_price = 18000.0

        # Random Walk mit kleiner positiver Drift (NQ tendiert langfristig hoch)
        returns = np.random.normal(loc=0.0001, scale=0.002, size=n_bars)
        prices = start_price * np.exp(np.cumsum(returns))

        # OHLC aus dem Close-Preis ableiten
        noise = np.random.normal(0, 3, n_bars)  # ±3 Punkte Intrabar-Bewegung
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n_bars, freq="1min"),
            "open":      prices + noise * 0.3,
            "high":      prices + np.abs(noise),
            "low":       prices - np.abs(noise),
            "close":     prices,
            "volume":    np.random.randint(100, 5000, n_bars),
        })

        return df

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Berechnet technische Indikatoren.
        Diese werden später vom Regime Classifier gebraucht.
        
        Berechnete Indikatoren:
        - ATR  (Average True Range): Misst die Volatilität
        - RSI  (Relative Strength Index): Misst Über/Unterkauft
        - ADX  (Average Directional Index): Misst Trendstärke
        - SMA  (Simple Moving Average): Gleitender Durchschnitt
        """
        # ATR berechnen (Volatilität)
        # True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                np.abs(df["high"] - df["close"].shift(1)),
                np.abs(df["low"] - df["close"].shift(1))
            )
        )
        df["atr"] = df["tr"].rolling(14).mean()  # 14-Perioden ATR

        # RSI berechnen (Momentum)
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)   # 1e-10 verhindert Division durch 0
        df["rsi"] = 100 - (100 / (1 + rs))

        # Einfaches ADX-Approximation (vereinfacht)
        df["adx"] = df["atr"].rolling(14).std() / df["atr"].rolling(14).mean() * 100

        # Moving Averages
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()

        # Volumen-Verhältnis (aktuelles Volumen / Durchschnittsvolumen)
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

        # Fehlende Werte am Anfang entfernen
        df = df.dropna().reset_index(drop=True)

        return df

    def _connect_live(self):
        """
        Verbindet mit dem Live-Broker.
        
        WICHTIG: Diese Funktion ist noch nicht implementiert!
        Für Live-Handel brauchst du:
        1. Einen Account bei Rithmic oder Interactive Brokers
        2. Die entsprechende Python-API-Bibliothek
        3. API-Zugangsdaten in config.py
        
        Für Rithmic: pip install rithmic
        Für IB:      pip install ibapi
        """
        log.warning("Live-Modus noch nicht implementiert!")
        log.warning("Nutze vorerst 'backtest' als Modus in config.py")
        raise NotImplementedError("Live-Verbindung noch nicht eingerichtet.")

    def get_latest_data(self) -> dict | None:
        """
        Gibt die aktuellen Marktdaten zurück.
        
        Im Backtest: Gibt die nächste Zeile aus der CSV zurück
        Im Live:     Holt die neuesten Daten vom Broker
        
        Rückgabe: Dictionary mit allen Marktdaten, oder None wenn fertig
        """
        if self.mode == "backtest":
            return self._get_backtest_bar()
        else:
            return self._get_live_data()

    def _get_backtest_bar(self) -> dict | None:
        """
        Gibt den nächsten Datenpunkt aus den historischen Daten zurück.
        Gibt None zurück wenn alle Daten verarbeitet wurden.
        """
        # Prüfen ob wir am Ende der Daten sind
        # Wir brauchen mindestens 100 vergangene Bars für die Analyse
        lookback = 100
        if self.current_index + lookback >= len(self.data):
            return None  # Keine weiteren Daten → Bot-Schleife beendet

        # Aktuelle Bar + die letzten 100 Bars für Berechnungen
        bar = self.data.iloc[self.current_index + lookback]
        history = self.data.iloc[self.current_index : self.current_index + lookback]

        # Orderbuch simulieren (für Backtest – in der Realität kommt das vom Broker)
        fake_orderbook = self._simulate_orderbook(bar["close"], bar["atr"])

        # Ergebnis-Dictionary zusammenbauen
        result = {
            "timestamp":    bar["timestamp"],
            "price":        bar["close"],
            "open":         bar["open"],
            "high":         bar["high"],
            "low":          bar["low"],
            "bid":          bar["close"] - 0.25,    # Bid = 1 Tick unter Close
            "ask":          bar["close"] + 0.25,    # Ask = 1 Tick über Close
            "volume":       int(bar["volume"]),
            "price_series": history["close"].values,  # Preishistorie (für Noise-Filter)
            "orderbook":    fake_orderbook,
            "trades":       [],  # Letzte Trades (im Backtest leer)
            "features": {       # Indikatoren für Regime-Classifier
                "atr":          bar["atr"],
                "rsi":          bar["rsi"],
                "adx":          bar["adx"],
                "volume_ratio": bar["volume_ratio"],
            },
        }

        self.current_index += 1  # Weiter zum nächsten Bar
        return result

    def _simulate_orderbook(self, mid_price: float, atr: float) -> dict:
        """
        Simuliert ein Orderbuch für Backtesting.
        
        In der Realität kommt das Orderbuch vom Broker (Level-2 Daten).
        Für Backtests simulieren wir es mit zufälligen Mengen.
        
        Parameter:
        - mid_price: Aktueller Marktpreis
        - atr:       Volatilität (beeinflusst Orderbuch-Spreads)
        """
        depth = 5  # 5 Bid + 5 Ask Level
        tick = 0.25  # NQ Tickgröße

        bids = []  # Kaufaufträge (unter aktuellem Preis)
        asks = []  # Verkaufsaufträge (über aktuellem Preis)

        for i in range(1, depth + 1):
            bid_price = mid_price - (i * tick)
            ask_price = mid_price + (i * tick)

            # Zufällige Mengen (näher am Marktpreis = mehr Volumen)
            bid_size = int(np.random.exponential(scale=50 / i))
            ask_size = int(np.random.exponential(scale=50 / i))

            bids.append({"price": bid_price, "size": bid_size})
            asks.append({"price": ask_price, "size": ask_size})

        return {"bids": bids, "asks": asks}

    def _get_live_data(self) -> dict:
        """
        Holt Live-Daten vom Broker.
        Noch nicht implementiert – für spätere Erweiterung.
        """
        raise NotImplementedError("Live-Daten noch nicht implementiert.")

    def send_order(self, order: dict):
        """
        Sendet eine Order an den Broker.
        
        Im Backtest: Nur Logging (kein echtes Trading)
        Im Live:     Echte Order an Broker senden
        
        Parameter:
        - order: Dictionary mit Order-Details (direction, size, price, etc.)
        """
        if self.mode == "backtest":
            log.info(f"[BACKTEST] Order simuliert: {order['direction']} "
                     f"{order['contracts']} Kontrakte @ {order['entry_price']:.2f}")
        else:
            # Hier würde der echte Broker-API-Call kommen
            raise NotImplementedError("Live-Order noch nicht implementiert.")
