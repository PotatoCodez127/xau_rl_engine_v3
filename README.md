# XAU RL Engine V2
A decoupled Deep Learning / Reinforcement Learning architecture for automated trading of Gold (XAUUSD), utilizing multi-timeframe structural analysis and DXY correlation.

## Architecture
- **The Oracle (Deep Learning):** A temporal attention network that acts as the pattern recognition engine, outputting directional probabilities based on price action and intermarket data.
- **The Manager (Reinforcement Learning):** A Soft Actor-Critic (SAC) agent that dynamically sizes positions and manages risk (Dynamic TP/SL) based on the Oracle's confidence and current portfolio drawdown.