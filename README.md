# Study Abroad Cost Explorer 🎓

A comprehensive analysis toolkit and interactive web application for exploring the **Cost of International Education** across major study-abroad destinations worldwide.

> streamlit website: https://study-abroad.streamlit.app/
> 
##  Features

###  Interactive Streamlit Web Application
- **Data Exploration Dashboard**: Interactive filters, visualizations, and statistics
- **ML-Powered Cost Prediction**: AI model predicting annual study costs
- **User-Friendly Interface**: Intuitive controls with preset buttons and helpful tooltips
- **Real-time Analysis**: Dynamic charts and cost breakdowns

###  Jupyter Notebooks
- **Cost Analysis Notebook**: Comprehensive EDA with visualizations
- **Model Training Notebook**: Machine learning pipeline for cost prediction

###  Dataset
- **International_Education_Costs.csv**: Curated data on tuition, living expenses, visa fees, and more
- **50+ countries** with detailed cost breakdowns

---

##  Quick Start

### Option 1: Run the Streamlit Web App (Recommended)

1. **Clone the repository**
   ```powershell
   git clone https://github.com/AdilShamim8/Study_Abroad.git
   cd Study_Abroad-main
   ```

2. **Create virtual environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```powershell
   streamlit run app.py
   ```

5. **Open in browser**
   - Local: http://localhost:8501
   - Network: http://192.168.0.101:8501

### Option 2: Run Jupyter Notebooks

```powershell
pip install jupyter pandas matplotlib seaborn
jupyter notebook
```

Then open:
- `Cost-of-studying-abroad.ipynb` for exploratory analysis
- `train_model.ipynb` for model training details

---

##  Dataset Overview

The dataset (`International_Education_Costs.csv`) includes:

| Column | Description |
|--------|-------------|
| **Country** | Destination country |
| **City** | Specific city location |
| **University** | Institution name |
| **Program** | Academic program/major |
| **Level** | Degree level (Bachelor's, Master's, PhD) |
| **Duration_Years** | Program length in years |
| **Tuition_USD** | Annual tuition fee in USD |
| **Living_Cost_Index** | Cost of living score (100 = baseline) |
| **Rent_USD** | Average monthly rent in USD |
| **Visa_Fee_USD** | Student visa application fee |
| **Insurance_USD** | Annual health insurance cost |
| **Exchange_Rate** | Local currency to USD rate |

**Target Variable**: `Estimated_Annual_Cost` = Tuition + Living Costs + Rent×12 + Visa + Insurance

---

##  Web App Features

### 1. Overview Section
- Quick statistics: countries, universities, programs
- Data quality metrics
- Missing values analysis

### 2. Data Exploration
- **Interactive Filters**: Country, Level, Program, Duration
- **Visualizations**:
  - Tuition and living cost distributions
  - Country-wise cost comparisons (Top 25)
  - Living cost vs. rent scatter plot with trendline
  - Global choropleth map of average costs
- **Statistics**: Descriptive stats for filtered data

### 3. Model Demonstration
- **User-Friendly Inputs**:
  -  Living Cost presets (Low/Medium/High)
  -  Currency quick-select buttons (USD, EUR, GBP, CAD, AUD, INR)
  - Visual indicators for cost levels
  - Helpful tooltips and examples
- **Predictions**:
  - AI-powered annual cost estimation
  - Cost breakdown with monthly estimates
  - Model performance metrics (MAE, R²)
- **Real-time Validation**: Input error checking

### 4. About Section
- Dataset documentation
- Model architecture details
- Feature explanations

---

##  Machine Learning Model

### Model Details
- **Algorithm**: Random Forest Regressor (best performer)
- **Preprocessing**: 
  - One-Hot Encoding for categorical features
  - Standard Scaling for numerical features
- **Features Used**: Country, Level, Program, Duration_Years, Living_Cost_Index, Exchange_Rate
- **Performance**: High R² score with low MAE

### Model Pipeline
The trained model (`model.pkl`) includes:
1. Preprocessing transformers (ColumnTransformer)
2. Trained Random Forest Regressor
3. Full end-to-end prediction pipeline

---

##  Technology Stack

### Web Application
- **Streamlit**: Interactive web framework
- **Plotly**: Interactive visualizations
- **Pandas & NumPy**: Data manipulation
- **scikit-learn**: Machine learning

### Analysis & Training
- **Jupyter**: Interactive notebooks
- **Matplotlib & Seaborn**: Static visualizations
- **scipy & statsmodels**: Statistical analysis

---

##  Project Structure

```
Study_Abroad-main/
├── app.py                              # Main Streamlit application
├── International_Education_Costs.csv   # Dataset
├── model.pkl                           # Trained ML model
├── requirements.txt                    # Python dependencies
├── train_model.ipynb                   # Model training notebook
├── Cost-of-studying-abroad.ipynb      # EDA notebook
├── utils/
│   └── data_model.py                  # Helper functions
├── tests/
│   └── test_app.py                    # Unit tests
├── venv/                              # Virtual environment
└── README.md                          # Documentation
```

---

##  Testing

Run tests to verify functionality:

```powershell
.\venv\Scripts\Activate.ps1
pip install pytest
pytest -q
```

Tests cover:
- Data loading and cleaning
- Model loading and prediction
- Metrics computation

---

##  Usage Examples

### Example 1: USA Master's Program
- **Country**: USA
- **Level**: Master
- **Program**: Computer Science
- **Duration**: 2 years
- **Living Cost**: 100 (Moderate)
- **Exchange Rate**: 1.0 (USD)

### Example 2: UK Graduate Program
- **Country**: United Kingdom
- **Level**: Master
- **Program**: Business Administration
- **Duration**: 1 year
- **Living Cost**: 140 (Expensive - London)
- **Exchange Rate**: 0.79 (GBP)

### Example 3: Germany Undergraduate
- **Country**: Germany
- **Level**: Bachelor
- **Program**: Engineering
- **Duration**: 3 years
- **Living Cost**: 75 (Affordable - Berlin)
- **Exchange Rate**: 0.92 (EUR)

---

##  Troubleshooting

### scikit-learn Version Mismatch
If you see version warnings:
```powershell
pip install --upgrade scikit-learn==1.6.1
```

### Port Already in Use
If port 8501 is busy:
```powershell
streamlit run app.py --server.port 8502
```

### Module Import Errors
Ensure virtual environment is activated:
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

##  Key Insights from Analysis

### Cost Patterns
- **Tuition vs. Living Costs**: Inverse relationship in some regions
- **Regional Variations**: Europe offers lower tuition, USA has higher costs
- **Program Impact**: STEM programs generally cost more than humanities

### Budget Planning
- **Low Budget**: Germany, France, Norway (€800-1,200/month)
- **Medium Budget**: Canada, Australia, Netherlands (€1,200-1,800/month)
- **High Budget**: USA, UK, Switzerland (€1,800-3,000/month)

### Hidden Costs
- Visa fees range from $50-500
- Health insurance: $500-3,000/year
- Exchange rate fluctuations can impact budgets by 10-20%

---

##  Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contribution
- Add more countries/universities
- Improve ML model accuracy
- Add scholarship data integration
- Implement cost comparison tools
- Add currency conversion API

---

## License

Licensed under the [License](License) file in the repository.

---

## Author

**Adil Shamim**

[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/adilshamim8)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/adilshamim8)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/adil_shamim8)

---

##  Acknowledgments

- UNESCO & OECD for education statistics
- Numbeo & Mercer for cost of living data
- scikit-learn & Streamlit communities
- All contributors and users

---

## Support

Having issues? Please:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Search existing issues on GitHub
3. Create a new issue with detailed information

---

**⭐ Star this repo if you find it helpful!**

Last Updated: November 14, 2025
