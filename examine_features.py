import pickle
import pandas as pd

def examine_feature_info():
    """Examine the feature_info.pkl file to understand training configuration."""
    
    with open('feature_info.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print("All keys in feature_info:")
    for k, v in data.items():
        print(f"{k}: {type(v)}")
        if isinstance(v, dict):
            print(f"  Dict contents: {v}")
        elif isinstance(v, list):
            print(f"  List length: {len(v)}, first few items: {v[:5]}")
        else:
            print(f"  Value: {v}")
        print()
    
    # Also check what columns are in the original CSV
    df = pd.read_csv('training_data.csv')
    print("CSV columns:")
    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")
    
    print(f"\nTotal CSV columns: {len(df.columns)}")

if __name__ == "__main__":
    examine_feature_info() 