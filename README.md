# Study_Abroad Dataset & Notebook 📊📚

A curated dataset and accompanying Jupyter notebook to analyze the **Cost of International Education** across major study-abroad destinations. This repository hosts:

* **Dataset**: `cost_of_international_education.csv` — key metrics on tuition, living expenses, visa fees, and scholarship availability.
* **Notebook**: `Cost_of_International_Education_Analysis.ipynb` — clean data, perform descriptive analysis, visualize cost patterns, and draw insights.


##  Overview 

This repository provides a comprehensive dataset and analysis tools to estimate the cost of studying abroad across various countries, cities, and universities. It aims to assist prospective international students in making informed decisions by offering detailed insights into tuition fees, living expenses, visa fees, insurance costs, and more.
---

## Dataset Overview

The dataset includes the following columns:

* **Country**: The country where the university is located.
* **City**: The specific city of the university.
* **University**: Name of the institution.
* **Program**: The academic program offered.
* **Level**: Degree level (e.g., Bachelor's, Master's, PhD).
* **Duration_Years**: Length of the program in years.
* **Tuition_USD**: Annual tuition fee in USD.
* **Living_Cost_Index**: An index representing the cost of living in the city.
* **Rent_USD**: Average monthly rent in USD.
* **Visa_Fee_USD**: Student visa application fee in USD.
* **Insurance_USD**: Annual health insurance cost in USD.
* **Exchange_Rate**: Local currency to USD exchange rate.
---

## 📈 Notebook Analysis

The accompanying Kaggle notebook provides exploratory data analysis (EDA) to help users understand:

* Cost comparisons across different countries and cities.
* Tuition fee distributions by program and degree level.
* Impact of living costs and rent on overall expenses.
* Visa and insurance fee variations.

  
##  Getting Started

### Prerequisites

* Python 3.8+ with `pandas`, `matplotlib`, and `seaborn`
* Jupyter Notebook or JupyterLab

```bash
pip install pandas matplotlib seaborn notebook
```

### Running the Notebook

1. **Clone the repo**

   ```bash
   git clone https://github.com/AdilShamim8/Study_Abroad.git
   cd Study_Abroad
   ```
2. **Launch Jupyter**

   ```bash
   Jupyter notebook notebooks/Cost_of_International_Education_Analysis.ipynb
   ```
3. **Explore Analysis**

   * Data cleaning and validation steps
   * Descriptive statistics for each cost component
   * Visualizations: bar charts, box plots, cost comparisons



##  Key Insights

* **Tuition vs. Living Costs**: Some countries have low tuition but high living expenses, and vice versa.
* **Visa Fees Impact**: Visa processing fees are a small fraction, but vary significantly.
* **Scholarship Availability**: Correlates moderately with overall cost.

(See the notebook for detailed charts and commentary.)

##  Contributing

Contributions are welcome! If you have suggestions for additional data points, improvements to the analysis, or other enhancements, please fork the repository and submit a pull request.

##  License 

Licensed under the [MIT License](LICENSE).

