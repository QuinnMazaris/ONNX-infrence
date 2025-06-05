# ONNX Inference Application

This project demonstrates how to perform ONNX model inference using C# and the ONNX Runtime library.

## Project Structure

- `OnnxModelApp/`: This directory contains the core C# library for ONNX model prediction.
- `TestApp/`: This directory contains a console application that demonstrates how to use the `OnnxModelApp` library.
- `Model/`: (Expected) This directory should contain your ONNX model file (e.g., `RandomForest_production.onnx`).

## Building and Running the Application

Follow these steps to build and run the application:

1.  **Navigate to the project root:**

    Open your terminal or command prompt and navigate to the root directory of this project (where `OnnxModelApp` and `TestApp` folders are located).

    ```bash
    cd "C:\Users\QuinnMazaris\Desktop\ONNX infrence"
    ```

2.  **Build the `OnnxModelApp` library:**

    This command compiles the core ONNX inference library.

    ```bash
    dotnet build OnnxModelApp
    ```

3.  **Build the `TestApp` console application:**

    This command compiles the example application that uses the `OnnxModelApp` library.

    ```bash
    dotnet build TestApp
    ```

4.  **Run the `TestApp`:**

    This command executes the `TestApp`, which will load the ONNX model and perform a prediction using the pre-defined input vector.

    ```bash
    dotnet run --project TestApp
    ```

    You should see output similar to this:

    ```
    Using model: .../RandomForest_production.onnx

    === Model Information ===
    Input metadata:
    ...
    Number of outputs: 2
    Output name: output_label, Type: ONNX_TYPE_TENSOR
    Output name: output_probability, Type: ONNX_TYPE_SEQUENCE
    Label result name: output_label
    Prediction Result: Good
    Confidence: 0.3746
    ```

## Input Data

The `TestApp` currently uses a hardcoded `float[] rawInputVector` in `TestApp/Program.cs` as input for the ONNX model. If your model requires specific ordering or preprocessing (like one-hot encoding), you are now responsible for preparing this `float[]` array before passing it to the `WeldPredictor.Predict()` method. 