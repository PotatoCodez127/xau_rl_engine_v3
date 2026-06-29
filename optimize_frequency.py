import os
import time
import pandas as pd
from m1_live_simulator import M1HighFidelitySimulator

def run_frequency_optimizer():
    # File Paths
    XAU = "data/raw/XAUUSDr_M1.csv"
    DXY = "data/raw/USDIndex_M1.csv"
    ORACLE = "models/oracle/best_oracle.pth"
    MANAGER = "models/manager/saved/wfa_43/best_model.zip"

    # Ensure logs directory exists for safety
    os.makedirs("logs", exist_ok=True)
    results_path = "logs/frequency_optimization_results.csv"
    results = []

    print("Starting Max Trades Per Day Optimization (1 to 20)...")

    for max_trades in range(1, 21):
        print(f"\n{'='*60}")
        print(f" OPTIMIZATION SWEEP: max_trades_per_day = {max_trades} ")
        print(f"{'='*60}")

        start_time = time.time()

        # Initialize the engine
        sim = M1HighFidelitySimulator(XAU, DXY, ORACLE, MANAGER)
        
        # Override the defaults
        sim.max_trades_per_day = max_trades
        sim.min_bars_between_trades = 4  # Keep the 1-hour cooldown static

        # Suppress the heavy 10k-tick progress prints to keep the Colab terminal clean
        # Execute the simulation and capture the yield dataframe
        yield_df = sim.run_simulation()

        # Process Results
        if not yield_df.empty:
            total_attempts = len(yield_df)
            passes = len(yield_df[yield_df['Result'] == 'PASSED'])
            fails = len(yield_df[yield_df['Result'] == 'FAILED'])
            winrate = (passes / total_attempts) * 100 if total_attempts > 0 else 0
            avg_trades = yield_df['Trades_Taken'].mean()
        else:
            total_attempts = passes = fails = winrate = avg_trades = 0

        execution_time = time.time() - start_time

        # Append to matrix
        results.append({
            "Max_Daily_Trades": max_trades,
            "Total_Challenges": total_attempts,
            "Passed": passes,
            "Failed": fails,
            "Winrate_Pct": round(winrate, 2),
            "Avg_Trades_Per_Challenge": round(avg_trades, 1),
            "Execution_Time_Sec": round(execution_time, 2)
        })

        # Save checkpoint iteratively to protect against Colab disconnects
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_path, index=False)
        print(f"--> Checkpoint saved to {results_path}")

    # Final Output and Ranking
    print("\n" + "="*60)
    print(" OPTIMIZATION COMPLETE: BEST CONFIGURATIONS ")
    print("="*60)
    
    # Sort by Winrate first, then by absolute volume of Passed challenges
    sorted_df = results_df.sort_values(by=["Winrate_Pct", "Passed"], ascending=[False, False])
    print(sorted_df.to_string(index=False))

if __name__ == "__main__":
    run_frequency_optimizer()