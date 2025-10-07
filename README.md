# ⚡ Wind Turbine Prediction using PyTorch

This repository contains a technical implementation of a **wind turbine performance prediction model** using **PyTorch**.  
The project trains a regression neural network on turbine sensor data to estimate key operational parameters such as **power output**, **rotor speed**, and **generator RPM**.

---

## 🧠 Project Overview

This notebook (`Vindturbin i Pytorch.ipynb`) demonstrates a complete workflow for building and training a machine learning model on wind turbine sensor data.  
The project includes:
- Loading and preprocessing raw CSV data
- Building a **fully connected feedforward neural network (FFNN)** in PyTorch
- Training, validation, and loss visualization
- Evaluation of model performance

---

## 🧩 Dataset

The dataset (`Wind_Turbine_Data.csv`) includes multiple turbine-related continuous features such as:

| Feature | Description |
|----------|--------------|
| `WT1 - Wind speed (m/s)` | Measured wind velocity |
| `WT1 - Power (kW)` | Generated electrical power |
| `WT1 - Rotor speed (RPM)` | Turbine rotor rotational speed |
| `WT1 - Generator RPM (RPM)` | Generator speed |
| `WT1 - Nacelle position (°)` | Orientation of turbine nacelle |
| `WT1 - Blade angle (pitch position) (°)` | Blade pitch control angle |

---

## ⚙️ Requirements

Install dependencies before running the notebook:

```bash
pip install torch torchvision torchaudio pandas numpy matplotlib scikit-learn
```

---

## 🧮 Model Architecture

The model is implemented as a **regression neural network** using PyTorch:

```python
import torch
import torch.nn as nn

class WindTurbineNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(WindTurbineNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
```

### Hyperparameters
```python
input_size = 10
hidden_size = 64
output_size = 1
learning_rate = 0.001
num_epochs = 500
batch_size = 32
```

---

## 🧰 Training Pipeline

```python
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
```

During training, both **training** and **validation losses** are monitored and plotted using `matplotlib` for convergence analysis.

---

## 📈 Evaluation

The model’s performance is evaluated using metrics such as:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score** (via `sklearn.metrics`)

Visualization of predictions vs. actual values helps assess regression accuracy.

```python
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
print(f"MSE: {mse:.4f}, R²: {r2:.4f}")
```

---

## 📊 Results

| Metric | Description | Result |
|---------|--------------|--------|
| Training Loss | MSE Loss during training | ↓ consistently |
| Validation Loss | Evaluated loss on validation set | Stabilized |
| R² Score | Fit quality | Good correlation between predictions and actual values |

---

## 🧪 Future Improvements

- Add dropout layers for better generalization
- Hyperparameter tuning via Optuna
- Expand dataset with external weather data (wind direction, temperature)
- Deploy model for real-time inference with **FastAPI** or **Flask**

---

## 📁 Repository Structure

```
├── Vindturbin i Pytorch.ipynb
├── Wind_Turbine_Data.csv
├── README.md
```

---

## 👤 Author

**Morteza Naseri**  
Developed using **PyTorch** for academic and research purposes in wind turbine performance prediction.

---

## 📜 License

This project is released under the **MIT License**.  
You may use and modify the code with proper attribution.

---
