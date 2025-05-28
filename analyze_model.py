import onnx
import pickle
import pandas as pd
import numpy as np

def analyze_model():
    # Load and analyze ONNX model
    print("Analyzing ONNX model...")
    model = onnx.load("RandomForest_production.onnx")
    
    print("\nModel Inputs:")
    for input in model.graph.input:
        print(f"Name: {input.name}")
        print(f"Type: {input.type.tensor_type.elem_type}")
        print("Shape:", end=" ")
        for dim in input.type.tensor_type.shape.dim:
            if dim.dim_value:
                print(dim.dim_value, end=" ")
            else:
                print("?", end=" ")
        print("\n")

    print("\nModel Outputs:")
    for output in model.graph.output:
        print(f"Name: {output.name}")
        print(f"Type: {output.type.tensor_type.elem_type}")
        print("Shape:", end=" ")
        for dim in output.type.tensor_type.shape.dim:
            if dim.dim_value:
                print(dim.dim_value, end=" ")
            else:
                print("?", end=" ")
        print("\n")

def analyze_features():
    # Load and analyze feature information
    print("\nAnalyzing feature information...")
    with open("feature_info.pkl", 'rb') as f:
        feature_info = pickle.load(f)
    print("\nFeature Information:")
    print(feature_info)

def analyze_training_data():
    # Load and analyze training data
    print("\nAnalyzing training data...")
    df = pd.read_csv("training_data.csv")
    print("\nTraining Data Info:")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())

if __name__ == "__main__":
    analyze_model()
    analyze_features()
    analyze_training_data() 