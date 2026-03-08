"""
╔══════════════════════════════════════════════════════════════════╗
║                    config.py – EINSTELLUNGEN                     ║
║         Hier stellst du ALLES ein, ohne anderen Code anfassen   ║
╚══════════════════════════════════════════════════════════════════╝

Das ist die wichtigste Datei für dich als Anfänger!
Alle Einstellungen des Bots sind hier zentral gesammelt.
Du musst NUR diese Datei anpassen – nichts anderes.

WICHTIG: Starte immer im BACKTEST-Modus ("mode": "backtest")
         bevor du auf Live-Daten wechselst!
"""

CONFIG = {

    # ─────────────────────────────────────────────────────────────
    # GRUNDEINSTELLUNGEN
    # ─────────────────────────────────────────────────────────────

    "symbol": "NQ",             # Handelsinstrument: "NQ" = Nasdaq Futures
                                # Alternativ: "ES" (S&P), "YM" (Dow), "RTY" (Russell)

    "mode": "backtest",         # "backtest" = auf historischen Daten testen (SICHER)
                                # "live"     = echter Handel (NUR wenn du bereit bist!)

    "data_path": "data/nq_ticks.csv",   # Pfad zu deinen historischen Tick-Daten
                                         # Du brauchst eine CSV-Datei mit OHLCV-Daten

    "loop_interval": 1.0,       # Sekunden zwischen jedem Bot-Zyklus
                                # 1.0 = jede Sekunde, 0.1 = 10x pro Sekunde
                                # Für Backtest: 0.0 (so schnell wie möglich)


    # ─────────────────────────────────────────────────────────────
    # LAYER 1: NOISE-FILTER EINSTELLUNGEN
    # ─────────────────────────────────────────────────────────────

    "noise_method": "kde",      # "kde" = Kernel Density Estimation (empfohlen für Start)
                                # "gmm" = Gaussian Mixture Model (komplexer, genauer)

    "kde_bandwidth": 0.5,       # Wie stark KDE glättet
                                # Kleiner = weniger Glättung (mehr Details)
                                # Größer = mehr Glättung (weniger Noise, aber träger)

    "gmm_components": 3,        # Anzahl der Verteilungen im GMM
                                # 2 = Signal + Noise
                                # 3 = Signal + kleiner Noise + großer Noise

    "max_noise_level": 0.7,     # Wenn Noise > dieser Wert → kein Trade
                                # 0.0 = kein Noise erlaubt, 1.0 = alles erlaubt
                                # Empfehlung: 0.6 - 0.8

    "lookback_ticks": 100,      # Wie viele vergangene Ticks für Noise-Berechnung


    # ─────────────────────────────────────────────────────────────
    # LAYER 2: REGIME CLASSIFIER EINSTELLUNGEN
    # ─────────────────────────────────────────────────────────────

    "n_regimes": 4,             # Anzahl verschiedener Markt-Zustände
                                # 4 empfohlen: trending_up, trending_down,
                                #              mean_reverting, high_volatility

    "regime_model": None,       # Pfad zu einem vortrainierten Modell
                                # None = neues Modell wird trainiert
                                # Beispiel: "models/regime_hmm.pkl"

    "regime_features": [        # Welche Indikatoren für Regime-Erkennung genutzt werden
        "atr",                  # Average True Range (Volatilität)
        "adx",                  # Average Directional Index (Trendstärke)
        "rsi",                  # Relative Strength Index
        "volume_ratio",         # Volumen vs. Durchschnitt
    ],

    "regime_lookback": 50,      # Wie viele Bars zurück für Regime-Berechnung


    # ─────────────────────────────────────────────────────────────
    # LAYER 3: ORDERFLOW / HEATMAP EINSTELLUNGEN
    # ─────────────────────────────────────────────────────────────

    "orderbook_depth": 10,      # Wie viele Preislevels vom Orderbuch analysiert werden
                                # 5  = nur die 5 besten Bid/Ask-Level
                                # 10 = tiefere Analyse (mehr Daten nötig)

    "orderflow_model": None,    # Pfad zu vortrainiertem CNN-Modell
                                # None = Einfache regelbasierte Analyse statt CNN

    "heatmap_history": 30,      # Wie viele Orderbuch-Snapshots für Heatmap genutzt werden
                                # Mehr = bessere Muster, aber mehr Speicher


    # ─────────────────────────────────────────────────────────────
    # LAYER 4: PROBABILITY ENGINE EINSTELLUNGEN
    # ─────────────────────────────────────────────────────────────

    "min_confidence": 0.65,     # Mindest-Wahrscheinlichkeit für einen Trade
                                # 0.65 = mindestens 65% Konfidenz nötig
                                # Höher = weniger aber bessere Trades

    "mc_simulations": 1000,     # Anzahl Monte Carlo Simulationen
                                # Mehr = genauer, aber langsamer
                                # 500-2000 ist ein guter Bereich

    # Gewichtung der einzelnen Signale (müssen sich auf 1.0 addieren!)
    "signal_weights": {
        "regime":     0.35,     # Wie stark das Regime-Signal gewichtet wird (35%)
        "orderflow":  0.40,     # Orderflow-Signal (40%) - wichtigstes Signal!
        "noise":      0.25,     # Noise-Level (25%)
    },


    # ─────────────────────────────────────────────────────────────
    # RISIKO-MANAGEMENT EINSTELLUNGEN
    # DAS IST DER WICHTIGSTE ABSCHNITT! Hier schützt du dein Kapital.
    # ─────────────────────────────────────────────────────────────

    "account_size": 10000,      # Kontogröße in USD
                                # Passe das an dein echtes Konto an!

    "max_risk_pct": 0.02,       # Maximales Risiko pro Trade = 1% des Kontos
                                # Bei $50.000 Konto = max. $500 Risiko pro Trade
                                # Anfänger: 0.005 (0.5%) bis 0.01 (1%)

    "max_daily_loss": 0.03,     # Maximaler Tagesverlust = 3% des Kontos
                                # Bei $50.000 = max. $1.500 Verlust pro Tag
                                # Dann stoppt der Bot automatisch!

    "stop_loss_atr_mult": 1.5,  # Stop-Loss = 1.5 × ATR unter/über Entry
                                # Größer = weiterer Stop (weniger Ausstopper, mehr Risiko)
                                # Kleiner = engerer Stop (mehr Ausstopper, weniger Risiko)

    "take_profit_ratio": 2.0,   # Take-Profit = 2× der Stop-Loss-Distanz
                                # (Risk/Reward-Ratio von 1:2)
                                # Empfehlung: mindestens 1.5 (also 1:1.5)

    "max_open_positions": 1,    # Maximale gleichzeitige Positionen
                                # Anfänger: 1 (immer nur eine Position gleichzeitig)


    # ─────────────────────────────────────────────────────────────
    # BROKER / DATENFEED EINSTELLUNGEN
    # (Nur relevant wenn du LIVE handelst)
    # ─────────────────────────────────────────────────────────────

    "broker": "paper",          # "paper"   = Simulierter Handel (kein echtes Geld)
                                # "rithmic" = Rithmic API (professionell)
                                # "cqg"     = CQG API (professionell)
                                # "ib"      = Interactive Brokers

    "rithmic_user": "",         # Dein Rithmic Benutzername (leer lassen wenn nicht genutzt)
    "rithmic_pass": "",         # Dein Rithmic Passwort
    "rithmic_server": "test",   # "test" = Testserver, "live" = Echtgeld!

    "tick_size": 0.25,          # Minimale Preisbewegung für NQ = 0.25 Punkte
    "tick_value": 5.0,          # Ein NQ-Tick = $5.00 Wert
    "contract_multiplier": 20,  # NQ: 1 Punkt = $20 (= 4 Ticks × $5)

}
