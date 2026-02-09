# Hybrid Quantum LSTM vs. Classical LSTM for Financial Time Series Prediction

The goal was to explores the application of Quantum Machine Learning (QML) in financial forecasting. Specifically, it compares the performance of a **Classical Long Short-Term Memory (LSTM)** network against a **Hybrid Quantum LSTM (QLSTM)** in predicting the 5-day future returns of Apple Inc. (AAPL) stock.

The project utilizes **TensorFlow** for the classical model and **PyTorch + PennyLane** for the quantum-classical hybrid model.

## 📌 Project Overview

Financial time series data is often noisy and non-stationary. While classical LSTMs are powerful, this experiment investigates whether replacing classical linear layers within an LSTM cell with **Variational Quantum Circuits (VQC)** offers any advantage regarding convergence speed, generalization, or directional accuracy.

### Key Objectives:
1.  Preprocess financial data with technical indicators.
2.  Train a standard Classical LSTM.
3.  Train a Hybrid Quantum LSTM (using PCA for dimensionality reduction).
4.  Compare RMSE (Root Mean Squared Error) and Directional Accuracy.

## 🛠️ Technologies & Dependencies

*   **Python 3.11+**
*   **Data Handling:** `pandas`, `numpy`, `datasets` (Hugging Face)
*   **Preprocessing:** `scikit-learn` (StandardScaler, PCA)
*   **Classical ML:** `tensorflow` (Keras)
*   **Quantum ML:** `pennylane`, `torch` (PyTorch)
*   **Visualization:** `matplotlib`

## 📊 Dataset & Preprocessing

*   **Source:** [Adilbai/stock-dataset](https://huggingface.co/datasets/Adilbai/stock-dataset) via Hugging Face.
*   **Asset:** AAPL (Apple Inc.).
*   **Features:**
    *   OHLCV (Open, High, Low, Close, Volume).
    *   Technical Indicators: SMA, EMA, MACD, RSI, Bollinger Bands, Volatility.
    *   Lag Features: Previous days' close, volume, and returns.
*   **Target:** `Future_Return_5d` (Percentage return over the next 5 days).
*   **Sequence Length:** 10 days.
*   **Dimensionality Reduction:** For the QLSTM, **PCA (Principal Component Analysis)** is applied to reduce the 38 input features down to 6 principal components to fit within the qubit capacity of the quantum circuit.

## 🧠 Model Architectures

### 1. Classical LSTM
*   **Framework:** TensorFlow/Keras.
*   **Structure:**
    *   LSTM Layer (64 units).
    *   Dropout (0.3).
    *   Dense Layer (32 units, ReLU).
    *   Output Layer (Linear).
*   **Optimization:** Adam Optimizer, MSE Loss.

### 2. Hybrid Quantum LSTM (QLSTM)
*   **Framework:** PyTorch + PennyLane.
*   **Concept:** A custom LSTM cell where the internal gates (Input, Forget, Cell, Output) typically calculated by linear transformations ($W \cdot x + b$) are replaced by **Variational Quantum Circuits (VQC)**.
*   **Quantum Circuit Structure:**
    *   **Embedding:** Angle Embedding (RY rotations) to encode classical data into quantum states.
    *   **Entanglement:** Layers of CNOT gates to create dependencies between qubits.
    *   **Measurement:** Expectation values in the Z-basis.
*   **Qubits Used:** 10 (Input size + Hidden size).

##  Experimental Results

| Metric | Classical LSTM | Quantum LSTM (QLSTM) |
| :--- | :--- | :--- |
| **Epochs to Converge** | ~30 | **1** (Rapid Convergence) |
| **Test RMSE** (Lower is better) | 0.0742 | **0.0615** |
| **Directional Accuracy** (Higher is better) | 48.11% | **54.81%** |

### Key Observations
*   **Convergence:** The QLSTM demonstrated extremely rapid convergence, reaching its optimal loss within the very first epoch.
*   **Accuracy:** The QLSTM outperformed the classical model in both error minimization (RMSE) and, more importantly, **Directional Accuracy** (>50% implies better than random chance).
*   **Overfitting:** The QLSTM showed signs of overfitting immediately after Epoch 1 (Test MSE increased while Train MSE decreased), suggesting the model is highly expressive even with fewer parameters.

## 🚀 How to Run
**Install Requirements:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib tensorflow torch pennylane datasets
    ```

