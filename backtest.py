import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from env.xau_dynamic_env import XAUDynamicEnv
from models.oracle.attention_net import TemporalAttentionOracle
from sklearn.preprocessing import StandardScaler


def precompute_probabilities(df: pd.DataFrame, oracle_path: str) -> pd.DataFrame:
    """Passes the raw data through the frozen Oracle to get the AI probabilities."""
    print("Loading Oracle...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exclude_cols = ["target", "time", "datetime", "date"]
    feature_cols = [
        c for c in df.columns if c not in exclude_cols and not c.startswith("env_")
    ]

    oracle = TemporalAttentionOracle(input_dim=len(feature_cols), seq_len=30).to(device)
    oracle.load_state_dict(torch.load(oracle_path, map_location=device))
    oracle.eval()

    print("Calculating AI Probabilities...")
    scaler = StandardScaler()
    raw_features = scaler.fit_transform(df[feature_cols].values)
    probs_list = np.zeros((len(df), 3))

    with torch.no_grad():
        for i in range(30, len(df)):
            window = raw_features[i - 30 : i]
            window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)
            logits = oracle(window_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            probs_list[i] = probs

    df["prob_hold"] = probs_list[:, 0]
    df["prob_long"] = probs_list[:, 1]
    df["prob_short"] = probs_list[:, 2]

    return df


def generate_report(journal_df: pd.DataFrame, equity_curve: list):
    """Calculates and prints professional trading statistics."""
    os.makedirs("logs", exist_ok=True)

    if len(journal_df) == 0:
        print("\nNo trades were taken during the backtest period.")
        return

    wins = journal_df[journal_df["PnL_$"] > 0]
    losses = journal_df[journal_df["PnL_$"] <= 0]

    total_trades = len(journal_df)
    winrate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0

    avg_win_usd = wins["PnL_$"].mean() if len(wins) > 0 else 0
    max_win_usd = wins["PnL_$"].max() if len(wins) > 0 else 0
    min_win_usd = wins["PnL_$"].min() if len(wins) > 0 else 0

    avg_loss_usd = losses["PnL_$"].mean() if len(losses) > 0 else 0
    max_loss_usd = losses["PnL_$"].min() if len(losses) > 0 else 0
    min_loss_usd = losses["PnL_$"].max() if len(losses) > 0 else 0

    avg_tp_mult = wins["Take_Profit_Mult"].mean() if len(wins) > 0 else 0
    max_tp_mult = wins["Take_Profit_Mult"].max() if len(wins) > 0 else 0
    min_tp_mult = wins["Take_Profit_Mult"].min() if len(wins) > 0 else 0

    avg_sl_mult = losses["Stop_Loss_Mult"].mean() if len(losses) > 0 else 0
    max_sl_mult = losses["Stop_Loss_Mult"].max() if len(losses) > 0 else 0
    min_sl_mult = losses["Stop_Loss_Mult"].min() if len(losses) > 0 else 0

    print("\n" + "=" * 40)
    print(" ⚜️ XAU RL V2 PERFORMANCE REPORT ⚜️")
    print("=" * 40)
    print(f"Total Trades Taken:   {total_trades}")
    print(f"Winrate:              {winrate:.2f}%")
    print(f"Final Equity:         ${equity_curve[-1]:.2f}")

    # Calculate True Drawdown from the normalized equity curve
    peak = 10000.0
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        if dd > max_dd:
            max_dd = dd
    print(f"Max Drawdown:         {max_dd * 100:.2f}%\n")

    print("--- TAKE PROFIT METRICS (Normalized $100 Risk) ---")
    print(
        f"Average Win:          +${avg_win_usd:.2f} (Avg TP Mult: {avg_tp_mult:.2f}x)"
    )
    print(
        f"Largest Win:          +${max_win_usd:.2f} (Max TP Mult: {max_tp_mult:.2f}x)"
    )
    print(
        f"Smallest Win:         +${min_win_usd:.2f} (Min TP Mult: {min_tp_mult:.2f}x)\n"
    )

    print("--- STOP LOSS METRICS (Normalized $100 Risk) ---")
    print(
        f"Average Loss:         ${avg_loss_usd:.2f} (Avg SL Mult: {avg_sl_mult:.2f}x)"
    )
    print(
        f"Largest Loss:         ${max_loss_usd:.2f} (Max SL Mult: {max_sl_mult:.2f}x)"
    )
    print(
        f"Smallest Loss:        ${min_loss_usd:.2f} (Min SL Mult: {min_sl_mult:.2f}x)"
    )
    print("=" * 40)

    # Chart Generation
    plt.figure(figsize=(12, 6))
    plt.plot(
        equity_curve,
        label="Equity Curve (Fixed $100 Risk)",
        color="#00ffcc",
        linewidth=2,
    )
    plt.fill_between(
        range(len(equity_curve)),
        equity_curve,
        min(equity_curve) * 0.99,
        color="#00ffcc",
        alpha=0.1,
    )

    plt.title("RL Agent Equity Curve (Out of Sample)", fontsize=16, color="white")
    plt.xlabel("Steps (15m Candles)", fontsize=12, color="white")
    plt.ylabel("Account Balance ($)", fontsize=12, color="white")
    plt.grid(color="#333333", linestyle="--", linewidth=0.5)

    ax = plt.gca()
    ax.set_facecolor("#1e1e1e")
    plt.gcf().patch.set_facecolor("#1e1e1e")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")

    plt.legend(facecolor="#333333", edgecolor="white", labelcolor="white")

    chart_path = "logs/equity_curve.png"
    plt.savefig(chart_path, bbox_inches="tight")
    print(f"\n📊 Equity Curve chart saved to: {chart_path}")

    journal_path = "logs/final_backtest_journal.csv"
    journal_df.to_csv(journal_path, index=False)
    print(f"📓 Detailed trade journal saved to: {journal_path}")


def run_backtest():
    print("=== XAU RL V2 Backtest Engine ===")

    # 1. Point to your standard processed features file
    features_path = "data/processed/labeled_features_15m.csv"
    oracle_path = "models/oracle/best_oracle.pth"

    # 2. Point to the final split generated within the 80% boundary
    manager_path = "models/manager/saved/wfa_43/best_model.zip"

    if not os.path.exists(manager_path):
        print(f"ERROR: Could not find final manager weights at {manager_path}")
        return

    # 3. Load the master dataset
    raw_df = pd.read_csv(features_path, index_col=0, parse_dates=True)

    # 4. Slice the unseen 20% (The exact data the firewall protected during training)
    test_size = int(len(raw_df) * 0.2)
    test_df = raw_df.iloc[-test_size:].copy()

    enriched_df = precompute_probabilities(test_df, oracle_path)

    print("Initializing SAC Manager...")
    env = XAUDynamicEnv(df=enriched_df)
    model = SAC.load(manager_path, env=env)

    print("Running Simulation...")
    obs, info = env.reset()
    terminated = False
    truncated = False

    journal = []
    equity_curve = [10000.0]

    # Ensure backtester matches env architectural constants
    cooldown_timer = 0
    oracle_threshold = 0.36  # Calibrated via parameter sweep
    friction_cost = 10.0
    cooldown_duration = 24

    while not (terminated or truncated):
        previous_balance = env.balance

        # Replace the manual action[0] directional logic inside the `while not (terminated or truncated):` loop
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        current_balance = env.balance
        
        # In the new env, if balance changes (excluding minor drift), a trade occurred
        trade_executed = abs((current_balance - previous_balance) - reward) > 0.1 # Approximation block

        # Because the env now perfectly mimics OOS, you can streamline the backtester:
        direction_val = 0
        if current_balance != previous_balance and env.bars_since_last_trade == 1:
            # We must derive what the environment did based on the Oracle state
            prob_long = env.df.loc[env.current_step - 1, "prob_long"]
            prob_short = env.df.loc[env.current_step - 1, "prob_short"]
            h4_trend = env.df.loc[env.current_step - 1, "h4_trend"]
            
            if prob_long > 0.35 and prob_long > prob_short and h4_trend > 0:
                direction_val = 1
            elif prob_short > 0.35 and prob_short > prob_long and h4_trend < 0:
                direction_val = 2

        # Access probabilities via the info dict passed from the new environment step
        prob_long = info.get("prob_long", 0.0)
        prob_short = info.get("prob_short", 0.0)

        # Mirror the environment logic to know if a trade actually executed
        trade_executed = False

        if cooldown_timer > 0:
            direction_val = 0
            cooldown_timer -= 1
        else:
            if direction_val == 1 and prob_long < oracle_threshold:
                direction_val = 0
            elif direction_val == 2 and prob_short < oracle_threshold:
                direction_val = 0

        if direction_val != 0:
            flat_risk_usd = 100.0
            sl_mult = info.get("sl_mult_used", 0.5)
            tp_mult = info.get("tp_mult_used", 5.0)

            if current_balance > previous_balance:
                rr_ratio = tp_mult / sl_mult if sl_mult > 0 else 1
                trade_pnl = (flat_risk_usd * rr_ratio) - friction_cost
            else:
                trade_pnl = -flat_risk_usd - friction_cost

            normalized_balance = equity_curve[-1] + trade_pnl
            equity_curve.append(normalized_balance)

            # Reset cooldown for the backtest loop tracker
            cooldown_timer = cooldown_duration

            journal.append(
                {
                    "Step": env.current_step
                    - 1,  # Account for the step increment inside env
                    "Action": "Long" if direction_val == 1 else "Short",
                    "Position_Size_%": "FLAT $100 RISK",
                    "Take_Profit_Mult": round(tp_mult, 2),
                    "Stop_Loss_Mult": round(sl_mult, 2),
                    "Oracle_Prob_Long": round(prob_long, 4),
                    "Oracle_Prob_Short": round(prob_short, 4),
                    "Friction_Cost": friction_cost,
                    "PnL_$": round(trade_pnl, 2),
                    "Equity": round(normalized_balance, 2),
                }
            )
        else:
            # Equity stays flat if holding or locked out
            equity_curve.append(equity_curve[-1])

    generate_report(pd.DataFrame(journal), equity_curve)


if __name__ == "__main__":
    run_backtest()
