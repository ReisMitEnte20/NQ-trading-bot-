"""
╔══════════════════════════════════════════════════════════════════╗
║           NQ FUTURES TRADING BOT - HAUPTPROGRAMM                ║
║    Basiert auf: KDE/GMM → HMM → CNN Orderflow → Bayesian NN    ║
╚══════════════════════════════════════════════════════════════════╝

Dieses ist die Hauptdatei (main.py), die du in PyCharm startest.
Sie verbindet alle Teile des Trading Bots miteinander.

STARTEN: Einfach diese Datei in PyCharm mit dem grünen Play-Button ▶ starten.
"""

import time
import logging
from datetime import datetime

# Unsere eigenen Module importieren (die anderen .py Dateien)
from data_feed import DataFeed
from noise_filter import NoiseFilter
from regime_classifier import RegimeClassifier
from orderflow_analyzer import OrderflowAnalyzer
from probability_engine import ProbabilityEngine
from risk_manager import RiskManager
from config import CONFIG

# ─────────────────────────────────────────────
# LOGGING SETUP
# Logging = der Bot schreibt auf, was er gerade tut.
# So kannst du in PyCharm in der Konsole alles mitlesen.
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),                        # Ausgabe in PyCharm Konsole
        logging.FileHandler("trading_bot.log"),         # Gleichzeitig in Datei speichern
    ]
)
log = logging.getLogger(__name__)


def main():
    """
    Die Hauptschleife des Trading Bots.
    Diese Funktion wird aufgerufen wenn du das Programm startest.
    
    ABLAUF:
    1. Alle Module initialisieren (einmalig beim Start)
    2. In einer Endlosschleife:
       a. Neue Marktdaten holen
       b. Noise herausfiltern
       c. Markt-Regime bestimmen
       d. Orderflow analysieren
       e. Handels-Wahrscheinlichkeit berechnen
       f. Entscheidung treffen (kaufen / verkaufen / warten)
       g. Risiko prüfen und ggf. Order senden
    """
    
    log.info("=" * 60)
    log.info("  NQ FUTURES TRADING BOT STARTET")
    log.info(f"  Zeit: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    log.info("=" * 60)

    # ── SCHRITT 1: Alle Komponenten initialisieren ──────────────
    log.info("Initialisiere alle Module...")

    # DataFeed: Holt die Marktdaten (Preise, Orderbook etc.)
    data_feed = DataFeed(
        symbol=CONFIG["symbol"],          # z.B. "NQ" für Nasdaq Futures
        mode=CONFIG["mode"],              # "live" oder "backtest"
        data_path=CONFIG["data_path"],    # Pfad zu historischen Daten (für Backtest)
    )

    # NoiseFilter: Filtert Rauschen aus den Tick-Daten (KDE/GMM)
    noise_filter = NoiseFilter(
        method=CONFIG["noise_method"],            # "kde" oder "gmm"
        bandwidth=CONFIG["kde_bandwidth"],        # Wie stark gefiltert wird
        n_components=CONFIG["gmm_components"],   # Anzahl GMM-Komponenten
    )

    # RegimeClassifier: Erkennt den aktuellen Marktzustand (HMM)
    regime_classifier = RegimeClassifier(
        n_regimes=CONFIG["n_regimes"],        # Anzahl verschiedener Markt-Zustände
        model_path=CONFIG["regime_model"],    # Gespeichertes Modell (wenn vorhanden)
    )

    # OrderflowAnalyzer: Analysiert das Orderbuch / Heatmap (CNN)
    orderflow_analyzer = OrderflowAnalyzer(
        depth=CONFIG["orderbook_depth"],       # Wie viele Preislevels vom Orderbuch
        model_path=CONFIG["orderflow_model"],  # Gespeichertes CNN-Modell
    )

    # ProbabilityEngine: Kombiniert alle Signale zu einer Wahrscheinlichkeit
    probability_engine = ProbabilityEngine(
        min_confidence=CONFIG["min_confidence"],   # Mindest-Konfidenz für einen Trade
        mc_simulations=CONFIG["mc_simulations"],   # Anzahl Monte Carlo Simulationen
    )

    # RiskManager: Prüft ob ein Trade sicher ist (Positionsgröße, Stop-Loss etc.)
    risk_manager = RiskManager(
        max_risk_per_trade=CONFIG["max_risk_pct"],   # Max. Risiko pro Trade (z.B. 1%)
        max_daily_loss=CONFIG["max_daily_loss"],      # Max. Tagesverlust bevor Stop
        account_size=CONFIG["account_size"],          # Kontogröße in USD
    )

    log.info("Alle Module erfolgreich geladen ✓")
    log.info(f"Symbol: {CONFIG['symbol']} | Modus: {CONFIG['mode'].upper()}")
    log.info("-" * 60)

    # ── SCHRITT 2: Hauptschleife ────────────────────────────────
    # Diese Schleife läuft solange der Bot aktiv ist.
    # Mit STRG+C in PyCharm kannst du den Bot sauber stoppen.
    
    iteration = 0  # Zählt wie viele Zyklen der Bot schon gemacht hat

    try:
        while True:
            iteration += 1
            log.info(f"\n{'─'*50}")
            log.info(f"  ZYKLUS #{iteration}")
            log.info(f"{'─'*50}")

            # ── a) Marktdaten holen ──────────────────────────
            # Holt den neuesten Tick / Preisbalken + Orderbuch
            market_data = data_feed.get_latest_data()

            if market_data is None:
                # Im Backtest: keine weiteren Daten = fertig
                log.info("Keine weiteren Daten. Bot beendet.")
                break

            log.info(f"Preis: {market_data['price']:.2f} | "
                     f"Bid: {market_data['bid']:.2f} | "
                     f"Ask: {market_data['ask']:.2f} | "
                     f"Volume: {market_data['volume']}")

            # ── b) Noise herausfiltern ───────────────────────
            # Unterscheidet echte Preisbewegungen von Rauschen
            filtered_price = noise_filter.filter(market_data["price_series"])
            noise_level = noise_filter.get_noise_level()

            log.info(f"Noise-Level: {noise_level:.3f} | "
                     f"Gefilterter Preis: {filtered_price:.2f}")

            # Wenn zu viel Noise → besser nicht handeln
            if noise_level > CONFIG["max_noise_level"]:
                log.warning(f"  ⚠ Noise zu hoch ({noise_level:.3f}) → überspringe Zyklus")
                time.sleep(CONFIG["loop_interval"])
                continue

            # ── c) Markt-Regime bestimmen ────────────────────
            # Aktueller Zustand: "trending_up", "trending_down",
            #                    "mean_reverting", "high_volatility"
            regime = regime_classifier.predict(
                prices=filtered_price,
                features=market_data["features"],   # ATR, ADX, etc.
            )

            log.info(f"Regime: [{regime['label'].upper()}] "
                     f"(Konfidenz: {regime['confidence']:.1%})")

            # ── d) Orderflow analysieren ─────────────────────
            # Analysiert das Orderbuch auf Muster:
            # Absorption, Spoofing, Iceberg-Orders
            orderflow_signal = orderflow_analyzer.analyze(
                orderbook=market_data["orderbook"],     # Level-2 Daten
                recent_trades=market_data["trades"],    # Letzte ausgeführte Trades
            )

            log.info(f"Orderflow: Richtung={orderflow_signal['direction']} | "
                     f"Stärke={orderflow_signal['strength']:.2f} | "
                     f"Muster={orderflow_signal['pattern']}")

            # ── e) Handels-Wahrscheinlichkeit berechnen ──────
            # Kombiniert alle Signale zu EINER Wahrscheinlichkeit
            trade_signal = probability_engine.calculate(
                regime=regime,
                orderflow=orderflow_signal,
                noise_level=noise_level,
                price=filtered_price,
            )

            log.info(f"Signal: {trade_signal['direction'].upper()} | "
                     f"Wahrscheinlichkeit: {trade_signal['probability']:.1%} | "
                     f"Konfidenzintervall: [{trade_signal['ci_low']:.1%} – {trade_signal['ci_high']:.1%}]")

            # ── f) Entscheidung: Handeln oder Warten? ────────
            if trade_signal["probability"] < CONFIG["min_confidence"]:
                log.info(f"  → Signal zu schwach. Warte auf bessere Gelegenheit.")
                time.sleep(CONFIG["loop_interval"])
                continue

            log.info(f"  ✓ Signal stark genug! Prüfe Risiko...")

            # ── g) Risiko prüfen und Order senden ────────────
            # Berechnet Positionsgröße, Stop-Loss, Take-Profit
            order = risk_manager.evaluate(
                signal=trade_signal,
                current_price=market_data["price"],
                regime=regime,
            )

            if order["approved"]:
                log.info(f"  ✓ ORDER FREIGEGEBEN:")
                log.info(f"    Richtung:      {order['direction'].upper()}")
                log.info(f"    Kontrakte:     {order['contracts']}")
                log.info(f"    Entry:         {order['entry_price']:.2f}")
                log.info(f"    Stop-Loss:     {order['stop_loss']:.2f}")
                log.info(f"    Take-Profit:   {order['take_profit']:.2f}")
                log.info(f"    Risiko/Trade:  ${order['risk_usd']:.0f}")

                # Im LIVE-Modus: Hier würde die echte Order gesendet werden
                # data_feed.send_order(order)  # ← Auskommentiert für Sicherheit!
                
            else:
                log.warning(f"  ✗ ORDER ABGELEHNT: {order['reason']}")

            # Warte bis zum nächsten Zyklus
            time.sleep(CONFIG["loop_interval"])

    except KeyboardInterrupt:
        # STRG+C wurde gedrückt → sauberes Beenden
        log.info("\n" + "=" * 60)
        log.info("  Bot wurde manuell gestoppt (STRG+C)")
        log.info("  Alle Positionen bitte manuell prüfen!")
        log.info("=" * 60)

    except Exception as e:
        # Unerwarteter Fehler → Bot stoppt und zeigt Fehlermeldung
        log.error(f"KRITISCHER FEHLER: {e}", exc_info=True)
        raise


# ─────────────────────────────────────────────
# Dieses if-Statement sorgt dafür, dass main()
# nur aufgerufen wird wenn DU die Datei startest –
# nicht wenn sie von einer anderen Datei importiert wird.
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
