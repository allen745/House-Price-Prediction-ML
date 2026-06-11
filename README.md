# House-Price-Prediction-ML
# 🏠 House Price Prediction

A Machine Learning project that uses **XGBoost Regression** to predict California house prices based on location, demographics, and housing features.

---

## 📌 Project Overview

This project builds a regression model to predict median house values in California districts using the California Housing dataset. It demonstrates end-to-end ML pipeline including data preprocessing, correlation analysis, model training, and evaluation.

---

## 📂 Dataset

- **File:** `housing.csv`
- **Source:** California Housing Dataset
- **Samples:** 20,640
- **Features:** 9 (after dropping target)
- **Target:** `median_house_value`
- **Missing Values:** `total_bedrooms` had 207 nulls → filled with mean

**Features:**
| Feature | Description |
|---------|-------------|
| longitude | Geographic longitude |
| latitude | Geographic latitude |
| housing_median_age | Median age of houses |
| total_rooms | Total rooms in district |
| total_bedrooms | Total bedrooms in district |
| population | District population |
| households | Number of households |
| median_income | Median income of households |
| ocean_proximity | Distance to ocean (dropped — categorical) |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Programming Language |
| NumPy | Numerical computations |
| Pandas | Data loading and analysis |
| Matplotlib | Data visualization |
| Seaborn | Correlation heatmap |
| Scikit-learn | Train-test split, metrics |
| XGBoost | Regression model |

---

## ⚙️ Project Workflow

1. **Data Loading** → Load California housing CSV
2. **EDA** → Shape, statistics, null check
3. **Data Cleaning** → Fill missing `total_bedrooms` with mean
4. **Correlation Analysis** → Seaborn heatmap
5. **Feature Engineering** → Drop categorical `ocean_proximity`
6. **Train-Test Split** → 80% training / 20% testing (`random_state=2`)
7. **Model Training** → XGBoost Regressor
8. **Evaluation** → R² Score, Mean Absolute Error
9. **Visualization** → Actual vs Predicted price scatter plot

---

## 📊 Model Performance

| Dataset | R² Score | Mean Absolute Error |
|---------|----------|-------------------|
| Training | 82.6% | $33,332 |
| Testing | 79.9% | $35,001 |

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/allen745/House-Price-Prediction-ML.git
cd House-Price-Prediction-ML
```

**2. Install dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

**3. Run the script**
```bash
python "House Price Prediction.py"
```

---

## 📁 Repository Structure

```
House-Price-Prediction-ML/
│
├── House Price Prediction.py   # Main ML script
├── housing.csv                  # Dataset
└── README.md                    # Project documentation
```

---

## 👨‍💻 Author

**Allen Christian** | Patent Holder
- 🎓 AI & Data Science Student — A.D. Patel Institute of Technology
- 💼 [LinkedIn](https://www.linkedin.com/in/allen-christian-708545409)
- 🌐 [Portfolio](https://allen745.github.io)
- 🐙 [GitHub](https://github.com/allen745)

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
