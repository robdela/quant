# Singularity
> Autonomous multi-strategy algorithmic portfolio for cryptocurrency markets.

Singularity is a modular, multi-subsystem trading system designed for systematic operation across centralized cryptocurrency exchanges 

For the strategies, I included a simple trend following strategy called "Quark" based on three indicators
---

## Architecture

```
Systems
├── Event Horizon       — Core long-term investment engine
├── Gravity Arena       — Altcoin relative strength tournament
├── Hyperlink System    — Active trading strategies
├── Saps                - Strategic portfolio system
└── Lagrange            — Systemic hedge & risk-off management

```

This repository contains 5 subsystems

---

## Subsystems

### Event Horizon
Long-term, conviction-based investment engine. Based on a relative strength portfolio system

- **Asset universe**: Large-cap crypto
- **Timeframe**: High-timeframe (daily)
- **Signal type**: Proprietary trend and regime detection
- **Exit logic**: Condition-based, no hard stop-loss
- **Allocation**: Primary capital pool

### Gravity Arena
Altcoin relative strength tournament system. Continuously scores a universe of 40 altcoins and concentrates capital in the strongest-ranked names while optionally shorting the weakest.

- **Asset universe**: 40 altcoins (USDT-margined, CEX)
- **Selection logic**: Multi-factor scoring (7-point system)
  - Longs: top n assets by composite score (7/7)


### Hyperlink System
Active trading strategy layer. 

- **Input your own trading strategy**: System has been built for short term trading
- **Risk**: Position-level sizing, portfolio circuit breakers

Here's an example of the performance using one of my proprietary strategies (Out of sample from 2025-06-01 YYYY-MM-DD)
<img width="1400" height="700" alt="Figure_1" src="https://github.com/user-attachments/assets/0fbfce01-6799-4ae8-a063-ca1260a3bae7" />


### Saps
Long/Short strategy based system

- **Logic**: Use the strategy on multiple coins at the same time with different weights
- **Mechanisms**: Chose between sharpe, sortino, omega and kelly for weights


###  Lagrange System
A dedicated long/short relative strength system based on Gravity Arena Manages the directional book, score-based selection, and rebalancing logic.
- Long : strongest 3 assets by relative scoring
- Short : weakest 3 assets by relative scoring



---
## Singularity
Singularity is a meta-portfolio system that combines three sub-systems (EventHorizon, GravityArena, and Lagrange) into a single equity curve using a regime-based allocation model.

### Current logic
RISK ON : EventHorizon = 64%, GravityArena = 16%, Lagrange = 20%
RISK OFF : EventHorizon = 0%, GravityArena = 0%, Lagrange = 20%

Of course the allocation logic can be changed to your likings and different systems, this is just a template to show what can be done




---
## Disclaimer

This repository documents the architecture and design of a proprietary trading system. Strategy logic, signal generation, and parameter files are not disclosed. Past performance does not guarantee future results. This is not financial advice.
