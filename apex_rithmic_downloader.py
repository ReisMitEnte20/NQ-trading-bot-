"""
╔══════════════════════════════════════════════════════════════════╗
║       apex_rithmic_downloader.py – APEX + RITHMIC DOWNLOADER    ║
║    Lädt historische NQ Daten über deinen Apex Rithmic Account   ║
╚══════════════════════════════════════════════════════════════════╝

VORAUSSETZUNGEN:
1. Apex Trader Funding Account mit Rithmic (du hast bereits einen ✓)
2. Deine Rithmic Zugangsdaten aus dem Apex Dashboard
3. Pakete installieren (im PyCharm Terminal):
   pip install async-rithmic pandas

WO FINDEST DU DEINE ZUGANGSDATEN?
→ Apex Dashboard: apextraderfunding.com → Login
→ Klicke auf "Rithmic and NinjaTrader Setup"
→ Dort siehst du deinen Rithmic Username und Passwort

WICHTIG FÜR APEX:
→ System:  "APEX"  (NICHT "Rithmic Test" oder "Rithmic 01"!)
→ Gateway: "Chicago Area"
→ Kein extra Conformance Test nötig – Apex hat das bereits gemacht!
"""

import asyncio
import pandas as pd
import os
import logging
from datetime import datetime, timedelta, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# !! HIER DEINE APEX RITHMIC ZUGANGSDATEN EINTRAGEN !!
#
# Wo findest du diese?
# 1. Gehe zu: apextraderfunding.com → Login
# 2. Klicke auf "Rithmic and NinjaTrader Setup"
# 3. Dort siehst du Username und Passwort
# ─────────────────────────────────────────────────────────────────

RITHMIC_USER     = "APEX-232987"   # z.B. "F-123456789-7"
RITHMIC_PASSWORD = "A@@jhP6wem7J"

# Diese Werte NICHT ändern – sie sind für Apex fix!
SYSTEM_NAME = "APEX"            # ← Muss "APEX" sein für Apex Accounts!
GATEWAY     = "Chicago Area"    # ← Apex nutzt Chicago als Gateway
APP_NAME    = "NQ_Trading_Bot"  # Frei wählbar
APP_VERSION = "1.0"


# ─────────────────────────────────────────────────────────────────
# DOWNLOAD EINSTELLUNGEN – Diese kannst du anpassen
# ─────────────────────────────────────────────────────────────────

# NQ Futures Symbol
# "NQ" = Nasdaq 100 Futures (du musst den aktuellen Contract angeben)
# Contract-Codes: H=März, M=Juni, U=September, Z=Dezember
# Aktuell (März 2026): NQM6 = Juni 2026 wäre der Front Month
SYMBOL   = "NQM6"     # ← Aktuellen Contract eintragen!
EXCHANGE = "CME"      # NQ handelt immer auf der CME

# Zeitraum für Download
# Tipp: Starte mit 1 Monat zum Testen!
START_DATE = datetime(2025, 12, 1, tzinfo=timezone.utc)
END_DATE   = datetime(2026,  2, 28, tzinfo=timezone.utc)

# Balken-Typ
BAR_MINUTES = 1       # 1 = 1-Minuten Bars (empfohlen)
                      # 5 = 5-Minuten (weniger Daten, schneller)
                      # 15 = 15-Minuten (für längerfristige Analyse)

# Ausgabe
OUTPUT_DIR  = "data"
OUTPUT_FILE = (
    f"data/nq_{BAR_MINUTES}min_"
    f"{START_DATE.strftime('%Y%m%d')}_"
    f"{END_DATE.strftime('%Y%m%d')}.csv"
)


# ─────────────────────────────────────────────────────────────────
# DOWNLOADER
# ─────────────────────────────────────────────────────────────────

class ApexRithmicDownloader:
    """
    Lädt historische NQ-Daten über deinen Apex Rithmic Account.

    Besonderheit für Apex:
    → System muss "APEX" sein (nicht "Rithmic Test")
    → Gateway ist "Chicago Area"
    → Kein eigener Conformance Test nötig
    """

    def __init__(self):
        self.client  = None
        self.all_bars = []

    async def connect(self) -> bool:
        """Verbindet mit Rithmic über Apex-Zugangsdaten."""
        try:
            from async_rithmic import RithmicClient, Gateway

            log.info("Verbinde mit Rithmic (Apex System)...")
            log.info(f"  User:    {RITHMIC_USER}")
            log.info(f"  System:  {SYSTEM_NAME}")
            log.info(f"  Gateway: {GATEWAY}")

            # Gateway-Objekt für Apex
            # async_rithmic kennt verschiedene Gateways
            # Für Apex nutzen wir den Chicago Gateway
            self.client = RithmicClient(
                user=        RITHMIC_USER,
                password=    RITHMIC_PASSWORD,
                system_name= SYSTEM_NAME,      # "APEX" ← wichtig!
                app_name=    APP_NAME,
                app_version= APP_VERSION,
                gateway=     Gateway.CHICAGO,  # Chicago Area für Apex
            )

            await self.client.connect()
            log.info("✓ Erfolgreich verbunden mit Rithmic (Apex)!")
            return True

        except ImportError:
            log.error("'async-rithmic' nicht installiert!")
            log.error("Installieren mit: pip install async-rithmic")
            return False

        except Exception as e:
            log.error(f"Verbindung fehlgeschlagen: {e}")
            log.error("")
            log.error("Häufige Fehlerursachen:")
            log.error("  1. Falscher Username/Passwort → Im Apex Dashboard prüfen")
            log.error("  2. Leerzeichen im Username/Passwort → Sorgfältig kopieren")
            log.error("  3. Market Data Session bereits offen")
            log.error("     → NinjaTrader oder RTrader Pro schließen!")
            log.error("     → Rithmic erlaubt nur 1 Market Data Session gleichzeitig")
            return False

    async def download(self):
        """
        Lädt Bars in Monats-Chunks herunter.

        Warum Chunks?
        → Rithmic begrenzt die Anzahl Bars pro einzelner Anfrage
        → Wir teilen den Zeitraum in kleinere Stücke auf
        → Jeder Chunk = 1 Monat Daten
        """
        log.info(f"Starte Download: {SYMBOL} | {BAR_MINUTES}-Min Bars")
        log.info(f"Zeitraum: {START_DATE.strftime('%d.%m.%Y')} → "
                 f"{END_DATE.strftime('%d.%m.%Y')}")

        current = START_DATE

        while current < END_DATE:
            chunk_end = min(current + timedelta(days=30), END_DATE)

            log.info(f"  Chunk: {current.strftime('%d.%m.%Y')} → "
                     f"{chunk_end.strftime('%d.%m.%Y')}...")

            try:
                # Historische Bars anfordern
                # async_rithmic nutzt die R|Protocol API im Hintergrund
                bars = await self.client.get_historical_time_bars(
                    symbol=            SYMBOL,
                    exchange=          EXCHANGE,
                    start_time=        current,
                    end_time=          chunk_end,
                    bar_type=          "MinuteBar",      # Minuten-Bars
                    bar_type_specifier= BAR_MINUTES,     # Wie viele Minuten
                )

                if bars:
                    self.all_bars.extend(bars)
                    log.info(f"  ✓ {len(bars):,} Bars "
                             f"(Gesamt: {len(self.all_bars):,})")
                else:
                    log.warning("  Keine Daten für diesen Chunk")
                    log.warning("  (Möglicherweise Wochenende/Feiertag)")

            except Exception as e:
                log.error(f"  Fehler bei Chunk: {e}")
                log.info("  Überspringe und mache weiter...")

            current = chunk_end
            # Kurze Pause zwischen Anfragen (Rithmic Rate Limit)
            await asyncio.sleep(1.5)

    async def disconnect(self):
        """Trennt die Verbindung sauber."""
        if self.client:
            try:
                await self.client.disconnect()
                log.info("Verbindung sauber getrennt ✓")
            except Exception:
                pass

    def save_csv(self):
        """
        Speichert die Daten als CSV – direkt kompatibel mit unserem Bot!

        Das Ergebnis passt sofort zu data_feed.py:
        Spalten: timestamp, open, high, low, close, volume
        """
        if not self.all_bars:
            log.error("Keine Daten zum Speichern!")
            return False

        log.info(f"Konvertiere {len(self.all_bars):,} Bars in CSV...")

        rows = []
        for bar in self.all_bars:
            # async_rithmic gibt Bar-Daten als Dictionary zurück
            # Spaltennamen können leicht variieren je nach Version
            # Wir prüfen mehrere mögliche Namen:

            def get_val(bar, *keys, default=0.0):
                """Holt einen Wert aus dem Bar-Dictionary."""
                for key in keys:
                    if key in bar and bar[key] is not None:
                        return bar[key]
                return default

            timestamp = get_val(bar,
                "bar_end_time", "timestamp", "time", "datetime",
                default=None
            )
            if timestamp is None:
                continue  # Bar ohne Zeitstempel überspringen

            rows.append({
                "timestamp": timestamp,
                "open":      get_val(bar, "open_price",  "open",   default=0.0),
                "high":      get_val(bar, "high_price",  "high",   default=0.0),
                "low":       get_val(bar, "low_price",   "low",    default=0.0),
                "close":     get_val(bar, "close_price", "close",  default=0.0),
                "volume":    get_val(bar, "volume",      "vol",    default=0),
            })

        if not rows:
            log.error("Keine gültigen Bars nach Verarbeitung!")
            return False

        # DataFrame erstellen
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Bereinigen
        df = df.drop_duplicates(subset=["timestamp"])   # Keine Duplikate
        df = df[df["close"] > 0]                        # Ungültige Preise raus
        df = df.dropna()

        # Ausgabeordner erstellen
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # CSV speichern
        df.to_csv(OUTPUT_FILE, index=False)

        # Zusammenfassung
        log.info("=" * 58)
        log.info(f"✓ GESPEICHERT!")
        log.info(f"  Datei:        {OUTPUT_FILE}")
        log.info(f"  Bars:         {len(df):,}")
        log.info(f"  Von:          {df['timestamp'].iloc[0].strftime('%d.%m.%Y %H:%M')}")
        log.info(f"  Bis:          {df['timestamp'].iloc[-1].strftime('%d.%m.%Y %H:%M')}")
        log.info(f"  Preis-Range:  {df['close'].min():.2f} – "
                 f"{df['close'].max():.2f}")
        log.info(f"  Ø Volumen:    {df['volume'].mean():.0f} Kontrakte/Bar")
        log.info("=" * 58)
        log.info("")
        log.info("NÄCHSTER SCHRITT – Trage in config.py ein:")
        log.info(f'  "data_path": "{OUTPUT_FILE}",')
        log.info(f'  "mode":      "backtest",')
        log.info("")
        log.info("Dann main.py starten – der Bot läuft mit echten Daten!")
        return True


# ─────────────────────────────────────────────────────────────────
# HAUPT-FUNKTION
# ─────────────────────────────────────────────────────────────────

async def main():
    log.info("=" * 58)
    log.info("  APEX TRADER FUNDING – RITHMIC DOWNLOADER")
    log.info(f"  Symbol:   {SYMBOL} @ {EXCHANGE}")
    log.info(f"  Zeitraum: {START_DATE.strftime('%d.%m.%Y')} – "
             f"{END_DATE.strftime('%d.%m.%Y')}")
    log.info(f"  Bars:     {BAR_MINUTES}-Minuten")
    log.info("=" * 58)

    # Zugangsdaten-Check
    if "DEIN_APEX" in RITHMIC_USER:
        log.error("══════════════════════════════════════════════")
        log.error("  ZUGANGSDATEN FEHLEN!")
        log.error("")
        log.error("  So findest du sie:")
        log.error("  1. apextraderfunding.com → Login")
        log.error("  2. 'Rithmic and NinjaTrader Setup' klicken")
        log.error("  3. Username + Passwort kopieren")
        log.error("  4. Oben in dieser Datei eintragen")
        log.error("══════════════════════════════════════════════")
        return

    # ⚠️ Wichtiger Hinweis für Apex User
    log.warning("WICHTIG: Schließe NinjaTrader/RTrader Pro bevor du startest!")
    log.warning("Rithmic erlaubt nur 1 Market Data Session gleichzeitig.")
    log.info("Starte in 3 Sekunden...")
    await asyncio.sleep(3)

    downloader = ApexRithmicDownloader()

    # Verbinden
    connected = await downloader.connect()
    if not connected:
        return

    try:
        # Daten herunterladen
        await downloader.download()

        # Als CSV speichern
        downloader.save_csv()

    finally:
        # Immer sauber trennen – wichtig!
        await downloader.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
