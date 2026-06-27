Act as a Senior Quantitative Deep Learning Architect. My objective is to achieve the absolute maximum risk-adjusted return (highest Calmar and Sortino ratios) for an algorithmic trading engine operating on Gold (XAUUSD) at the 15-minute timeframe. 

I have permission to make both slight parametric tweaks and magnificent architectural overhauls. 

Here is the current state of the engine (V3.2), which currently yields a +55% Out-Of-Sample net return:
1. Data Pipeline: 15m data augmented with fractional differentiation (d=0.45), 14-period ATR normalization, and Higher Timeframe (H4 and H1) macro vectors injected without look-ahead bias.
2. Phase A (The Oracle): A PyTorch Temporal Attention network predicting categorical momentum direction [Hold, Long, Short] over a 30-period window. It is evaluated using an Asymmetric Minority Threshold (Relative Conviction) rather than a Softmax absolute threshold.
3. Phase B (The RL Manager): A Soft Actor-Critic (SAC) continuous agent. It has been stripped of directional autonomy (Master-Slave architecture) and is only queried for volumetric sizing and dynamic TP/SL placement. 
4. Risk Mechanics: Dynamic 1.5% equity compounding. The action space is mathematically floored to guarantee a Take Profit between 1.0x and 3.0x of the Stop Loss to prevent risk inversion.
5. Execution Filter: A hard quantitative gatekeeper that physically blocks the Oracle's signals if they counter the H4 macro EMA trend.
6. Validation: Sequential Walk-Forward Analysis (WFA) across 44 splits with a 20% Out-Of-Sample holdout.

Your task is to compile a comprehensive, multi-phase master plan to radically increase the yield and predictive accuracy of this engine. 

Please structure your proposals across the following domains, providing the theoretical justification and the specific implementation path for each:
1. Architectural Overhauls: Should we replace the Temporal Attention Oracle with a Time-Series Transformer (e.g., PatchTST) or a State Space Model (e.g., Mamba)? Should the SAC Manager be swapped for Proximal Policy Optimization (PPO) or Distributional RL?
2. Feature Engineering: What advanced alternative data proxies (e.g., synthetic order flow, limit order book approximations, options skew proxies) can we derive purely from OHLCV data to feed the Oracle?
3. Reward Shaping & Loss Functions: How can we adjust the Focal Loss of the Oracle or the reward topography of the RL agent to better capture tail-end momentum distributions?
4. Execution & Sizing: Are there advanced Kelly Criterion adaptations or volatility-scaling methods that outperform flat 1.5% compounding?

Do not hold back on complexity. Provide the absolute best theoretical and practical pathways to maximize this system's alpha.