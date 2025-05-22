from src import preprocessing as prep

def main():
    # Load and preprocess data
    df = prep.load_data("data/Walmart.csv")
    df = prep.handle_missing_values(df)
    df = prep.preprocess_features(df)

if __name__ == "__main__":
    main()
