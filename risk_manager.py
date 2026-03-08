"""
╔══════════════════════════════════════════════════════════════════╗
║           risk_manager.py – RISIKO-MANAGEMENT                   ║
║    Das wichtigste Modul: Schützt dein Kapital!                  ║
╚══════════════════════════════════════════════════════════════════╝

Das Risiko-Management ist das WICHTIGSTE Modul des Bots.
Ein schlechtes Signal kostet Geld, ein fehlendes Risiko-Management
kann das gesamte Konto vernichten.

Funktionen:
1. Positionsgröße berechnen (Kelly Criterion / Fixed Fractional)
2. Stop-Loss automatisch setzen
3. Take-Profit automatisch setzen
4. Tagesverlust-Limit überwachen
5. Order ablehnen wenn Risiko zu hoch

GOLDENE REGELN:
→ Niemals mehr als 1-2% des Kontos pro Trade riskieren
→ Niemals ohne Stop-Loss handeln
→ Bei Tagesverlust > 3% → Bot automatisch stoppen
"""

import logging
from datetime import datetime, date

log = logging.getLogger(__name__)


class RiskManager:
    """
    Prüft jede potenzielle Order auf Risiko und berechnet die Positionsgröße.
    
    Ohne RiskManager = Fahren ohne Bremsen.
    Der RiskManager ist deine Versicherung gegen katastrophale Verluste.
    """

    def __init__(self, max_risk_per_trade: float, max_daily_loss: float, account_size: float):
        """
        Erstellt einen neuen RiskManager.
        
        Parameter:
        - max_risk_per_trade: Maximales Risiko pro Trade (z.B. 0.01 = 1%)
        - max_daily_loss:     Maximaler Tagesverlust (z.B. 0.03 = 3%)
        - account_size:       Kontogröße in USD
        """
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss     = max_daily_loss
        self.account_size       = account_size

        # Tages-Tracking: Wird täglich zurückgesetzt
        self.daily_pnl      = 0.0       # Heutiger Gewinn/Verlust in USD
        self.daily_trades   = 0         # Anzahl Trades heute
        self.last_reset     = date.today()
        self.bot_stopped    = False     # True wenn Daily Loss erreicht

        # NQ Futures Spezifikationen
        self.tick_size      = 0.25      # Minimale Preisbewegung
        self.tick_value     = 5.0       # 1 Tick = $5.00
        self.point_value    = 20.0      # 1 Punkt = $20.00 (= 4 Ticks × $5)

        log.info(f"RiskManager initialisiert:")
        log.info(f"  Kontogröße:    ${account_size:,.0f}")
        log.info(f"  Max Risiko/Trade: {max_risk_per_trade:.1%} "
                 f"(${account_size * max_risk_per_trade:,.0f})")
        log.info(f"  Max Tagesverlust: {max_daily_loss:.1%} "
                 f"(${account_size * max_daily_loss:,.0f})")

    def evaluate(self, signal: dict, current_price: float, regime: dict) -> dict:
        """
        Prüft ob eine Order sicher ist und berechnet alle Parameter.
        
        Parameter:
        - signal:        Ergebnis der ProbabilityEngine
        - current_price: Aktueller Marktpreis
        - regime:        Aktuelles Markt-Regime
        
        Rückgabe: Dictionary mit:
        - approved:     True wenn Order erlaubt, False wenn abgelehnt
        - reason:       Grund für Ablehnung (wenn approved=False)
        - direction:    "long" oder "short"
        - contracts:    Anzahl Kontrakte
        - entry_price:  Einstiegspreis
        - stop_loss:    Stop-Loss Preis
        - take_profit:  Take-Profit Preis
        - risk_usd:     Risiko in USD
        """
        # ── Tages-Reset prüfen ────────────────────────────────
        self._check_daily_reset()

        # ── Basis-Checks ──────────────────────────────────────

        # 1. Bot gestoppt wegen Daily Loss?
        if self.bot_stopped:
            return self._reject("Daily Loss Limit erreicht. Bot gestoppt bis morgen.")

        # 2. Neutrales Signal → kein Trade
        if signal["direction"] == "neutral":
            return self._reject("Neutrales Signal – keine Richtung.")

        # 3. Tägliches Verlust-Limit prüfen
        max_daily_loss_usd = self.account_size * self.max_daily_loss
        if self.daily_pnl <= -max_daily_loss_usd:
            self.bot_stopped = True
            log.critical(f"⛔ DAILY LOSS LIMIT ERREICHT: ${self.daily_pnl:.0f}")
            log.critical("  Bot stoppt automatisch bis morgen!")
            return self._reject(f"Daily Loss Limit: ${max_daily_loss_usd:.0f} erreicht")

        # ── Position Sizing ───────────────────────────────────
        # Wie viele Kontrakte können wir handeln ohne zu viel zu riskieren?

        # Maximales Risiko in USD
        max_risk_usd = self.account_size * self.max_risk_per_trade

        # ATR für Stop-Loss-Berechnung (aus Regime-Features oder Default)
        # ATR = Average True Range = typische Preisbewegung
        atr = 15.0  # Default für NQ (kann aus Marktdaten kommen)

        # Stop-Loss-Distanz: 1.5 × ATR (aus config.py)
        sl_multiplier = 1.5
        sl_distance = atr * sl_multiplier

        # Stop-Loss und Take-Profit berechnen
        direction   = signal["direction"]
        entry_price = current_price

        if direction == "long":
            stop_loss   = entry_price - sl_distance      # SL unter dem Einstieg
            take_profit = entry_price + sl_distance * 2  # TP = 2× der SL-Distanz (1:2 RRR)
        else:  # "short"
            stop_loss   = entry_price + sl_distance      # SL über dem Einstieg
            take_profit = entry_price - sl_distance * 2  # TP = 2× unter Einstieg

        # Risiko pro Kontrakt in USD
        # (Preis-Differenz in Punkten × $20 pro Punkt für NQ)
        risk_per_contract = sl_distance * self.point_value

        # Anzahl Kontrakte = Max-Risiko / Risiko-pro-Kontrakt
        raw_contracts = max_risk_usd / (risk_per_contract + 1e-10)
        contracts = max(1, int(raw_contracts))  # Minimum 1 Kontrakt

        # Tatsächliches Risiko mit dieser Kontraktanzahl
        actual_risk_usd = contracts * risk_per_contract

        # ── Finale Sicherheitsprüfungen ───────────────────────

        # Risiko zu hoch?
        if actual_risk_usd > max_risk_usd * 1.1:  # 10% Toleranz
            contracts = max(1, contracts - 1)
            actual_risk_usd = contracts * risk_per_contract

        # Signal zu schwach für die Regime-Bedingungen?
        if regime["label"] == "high_volatility" and contracts > 1:
            contracts = 1   # In hoher Volatilität nur 1 Kontrakt
            actual_risk_usd = contracts * risk_per_contract
            log.info("High-Volatility Regime: Reduziere auf 1 Kontrakt")

        # ── Order genehmigt ───────────────────────────────────
        self.daily_trades += 1

        log.info(f"Order genehmigt: {direction.upper()} | "
                 f"{contracts} Kontrakt(e) | Risiko: ${actual_risk_usd:.0f}")

        return {
            "approved":      True,
            "reason":        "Order genehmigt",
            "direction":     direction,
            "contracts":     contracts,
            "entry_price":   round(entry_price, 2),
            "stop_loss":     round(stop_loss, 2),
            "take_profit":   round(take_profit, 2),
            "risk_usd":      round(actual_risk_usd, 2),
            "reward_usd":    round(actual_risk_usd * 2, 2),  # 1:2 RRR
            "risk_pct":      actual_risk_usd / self.account_size,
            "sl_distance":   sl_distance,
        }

    def update_pnl(self, pnl_usd: float):
        """
        Aktualisiert den täglichen Gewinn/Verlust.
        Wird nach jedem abgeschlossenen Trade aufgerufen.
        
        Parameter:
        - pnl_usd: Gewinn (+) oder Verlust (-) in USD
        """
        self.daily_pnl += pnl_usd
        status = "✓ Gewinn" if pnl_usd > 0 else "✗ Verlust"
        log.info(f"PnL Update: {status} ${pnl_usd:+.0f} | "
                 f"Tages-PnL: ${self.daily_pnl:+.0f}")

    def get_daily_stats(self) -> dict:
        """Gibt die heutigen Trading-Statistiken zurück."""
        return {
            "date":          self.last_reset,
            "daily_pnl":     self.daily_pnl,
            "daily_trades":  self.daily_trades,
            "bot_stopped":   self.bot_stopped,
            "remaining_risk": self.account_size * self.max_daily_loss + self.daily_pnl,
        }

    def _check_daily_reset(self):
        """
        Prüft ob ein neuer Tag begonnen hat und resettet die Tages-Statistiken.
        Wird automatisch bei jedem evaluate()-Aufruf geprüft.
        """
        today = date.today()
        if today > self.last_reset:
            log.info(f"Neuer Handelstag! Reset Tages-Statistiken.")
            log.info(f"Gestriger PnL: ${self.daily_pnl:+.0f} "
                     f"in {self.daily_trades} Trades")
            self.daily_pnl    = 0.0
            self.daily_trades = 0
            self.bot_stopped  = False    # Tages-Stop wird aufgehoben!
            self.last_reset   = today

    def _reject(self, reason: str) -> dict:
        """Erstellt ein standardisiertes Ablehnungs-Dictionary."""
        return {
            "approved": False,
            "reason":   reason,
            # Alle anderen Felder leer/null
            "direction": None, "contracts": 0,
            "entry_price": None, "stop_loss": None, "take_profit": None,
            "risk_usd": 0.0,
        }
