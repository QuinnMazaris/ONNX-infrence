# ONNX Inference Application

This repository contains two .NET 8.0 projects:
- `OnnxModelApp`: A class library containing the ONNX model inference logic.
- `TestApp`: A console application that uses `OnnxModelApp` to perform predictions.

## Prerequisites

- [.NET 8.0 SDK](https://dotnet.microsoft.com/download/dotnet/8.0) installed.
- The ONNX model file `RandomForest_production.onnx`

## Build and Run Instructions

Navigate to the root directory of the application (where `ONNX infrence` is located) in your terminal.

### Build

To build the `TestApp` project and its dependencies, run the following command:

```bash
dotnet build TestApp
```

This will compile both `OnnxModelApp` and `TestApp`.

### Run

After successfully building and ensuring `RandomForest_production.onnx` is in the correct output directory, you can run the `TestApp` using:

```bash
dotnet run --project TestApp
```

Alternatively, you can navigate to the output directory (e.g., `TestApp/bin/Debug/net8.0/`) and run the executable directly:

```bash
cd TestApp/bin/Debug/net8.0/
./TestApp
```

Replace `Debug` with `Release` if you built in Release configuration.
