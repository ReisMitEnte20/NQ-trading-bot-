# 🤖 NQ Futures Trading Bot

Ein ML-gestützter Trading Bot für Nasdaq Futures (NQ)
basierend auf: KDE/GMM → HMM → Orderflow → Bayesian Ensemble

---

## 📁 Datei-Struktur

```
trading_bot/
│
├── main.py                 ← HIER STARTEN! (Hauptprogramm)
├── config.py               ← ALLE Einstellungen (nur diese Datei anpassen!)
├── requirements.txt        ← Benötigte Python-Pakete
│
├── data_feed.py            ← Layer 0: Marktdaten holen
├── noise_filter.py         ← Layer 1: KDE/GMM Noise-Filter
├── regime_classifier.py    ← Layer 2: HMM Regime-Erkennung
├── orderflow_analyzer.py   ← Layer 3: Orderbuch-Analyse
├── probability_engine.py   ← Layer 4: Signal-Kombination
├── risk_manager.py         ← Layer 5: Risiko & Position Sizing
│
├── data/                   ← Hier kommen deine historischen Daten rein
│   └── nq_ticks.csv        ← (selbst herunterladen oder generieren)
│
└── models/                 ← Hier werden trainierte Modelle gespeichert
    └── regime_hmm.pkl      ← (wird automatisch erstellt)
```

---

## 🚀 Setup in PyCharm (Schritt für Schritt)

### Schritt 1: PyCharm öffnen
- Öffne PyCharm
- Öffne den `trading_bot/` Ordner als Projekt

### Schritt 2: Python-Pakete installieren
1. Klicke unten auf den **"Terminal"** Tab in PyCharm
2. Tippe folgenden Befehl und drücke Enter:
```bash
pip install -r requirements.txt
```
3. Warte bis alle Pakete installiert sind (kann 2-5 Minuten dauern)

### Schritt 3: Bot starten
1. Öffne die Datei `main.py` in PyCharm
2. Klicke auf den **grünen Play-Button ▶** oben rechts
3. Der Bot startet im Backtest-Modus mit Demo-Daten

---

## ⚙️ Konfiguration (config.py)

**Du musst NUR die `config.py` Datei anpassen!**

Die wichtigsten Einstellungen:

| Einstellung | Was es macht | Empfehlung |
|---|---|---|
| `mode` | "backtest" oder "live" | Immer mit "backtest" starten! |
| `account_size` | Deine Kontogröße in USD | Dein echtes Kapital |
| `max_risk_pct` | Max. Risiko pro Trade | 0.01 (= 1%) für Anfänger |
| `max_daily_loss` | Max. Tagesverlust | 0.03 (= 3%) |
| `min_confidence` | Mindest-Signalstärke | 0.65 (65%) |

---

## 📊 Wie lese ich die Ausgabe?

Wenn du den Bot startest, siehst du in der PyCharm-Konsole:

```
10:23:41  [INFO]  ══════════════════════════════════════════
10:23:41  [INFO]    NQ FUTURES TRADING BOT STARTET
10:23:41  [INFO]  ══════════════════════════════════════════
10:23:42  [INFO]  ZYKLUS #1
10:23:42  [INFO]  Preis: 18045.50 | Bid: 18045.25 | Ask: 18045.75
10:23:42  [INFO]  Noise-Level: 0.312 | Gefilterter Preis: 18045.20
10:23:42  [INFO]  Regime: [TRENDING_UP] (Konfidenz: 78.3%)
10:23:42  [INFO]  Orderflow: Richtung=long | Stärke=0.71 | Muster=imbalance
10:23:42  [INFO]  Signal: LONG | Wahrscheinlichkeit: 72.4% | [65.1% – 79.8%]
10:23:42  [INFO]    ✓ Signal stark genug! Prüfe Risiko...
10:23:42  [INFO]    ✓ ORDER FREIGEGEBEN:
10:23:42  [INFO]      Richtung:    LONG
10:23:42  [INFO]      Kontrakte:   1
10:23:42  [INFO]      Entry:       18045.50
10:23:42  [INFO]      Stop-Loss:   18022.50
10:23:42  [INFO]      Take-Profit: 18091.50
10:23:42  [INFO]      Risiko/Trade: $460
```

---

## ⚠️ Wichtige Warnungen

> **NIEMALS mit echtem Geld starten bevor du:**
> 1. Mindestens 3 Monate Backtest-Ergebnisse analysiert hast
> 2. Den Code vollständig verstehst
> 3. Mit einem Demo-Konto getestet hast
> 4. Die Risiken von Futures-Trading kennst

---

## 🔧 Nächste Schritte (Erweiterungen)

Wenn du bereit für mehr bist:

1. **Echte historische Daten**: NQ-Tick-Daten von NinjaTrader oder Rithmic exportieren
2. **CNN für Orderflow**: `torch` installieren und CNN in `orderflow_analyzer.py` aktivieren
3. **Live-Verbindung**: Rithmic oder Interactive Brokers API anbinden
4. **Backtesting-Engine**: Performance-Metriken berechnen (Sharpe Ratio, Max Drawdown)
5. **Dashboard**: Web-Interface mit Plotly Dash bauen

---

## 📞 Fehlerbehebung

**`ModuleNotFoundError: No module named 'hmmlearn'`**
→ `pip install hmmlearn` im Terminal ausführen

**`FileNotFoundError: data/nq_ticks.csv`**
→ Normal! Bot nutzt automatisch Demo-Daten. Echte Daten später hinzufügen.

**Bot reagiert nicht auf STRG+C**
→ Klicke auf das rote **Stop-Symbol ■** in PyCharm
