import os

from src import preprocessing as prep
from src.random_forest import train_random_forest
from src.lstm_model import train_lstm


def main():
    os.makedirs("outputs", exist_ok=True)

    # Load and preprocess Walmart data (retained for backwards compatibility)
    df = prep.load_data("data/Walmart.csv")
    df = prep.handle_missing_values(df)
    df = prep.preprocess_features(df)

    # --- Part B: Random Forest (predict time_slot) ---
    print("\n" + "=" * 60)
    print("PART B – Random Forest: predict time_slot")
    print("=" * 60)
    train_random_forest("data/customer_model_table_dirty.csv")

    # --- Part C: LSTM (predict next purchase channel) ---
    print("\n" + "=" * 60)
    print("PART C – LSTM: predict next purchase channel")
    print("=" * 60)
    train_lstm("data/customer_transactions_dirty.csv")


if __name__ == "__main__":
    main()
