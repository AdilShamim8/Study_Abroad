# Study_Abroad Dataset & Notebook 📊📚

A curated dataset and accompanying Jupyter notebook to analyze the **Cost of International Education** across major study-abroad destinations. This repository hosts:

* **Dataset**: `cost_of_international_education.csv` — key metrics on tuition, living expenses, visa fees, and scholarship availability.
* **Notebook**: `Cost_of_International_Education_Analysis.ipynb` — clean data, perform descriptive analysis, visualize cost patterns, and draw insights.

---

## 🔍 Overview 

International study can be expensive. This dataset captures country-level cost components for students planning to study abroad, enabling data-driven decisions on destination choices.

---

## 📝 Dataset Description 

| Column                  | Description                                 |
| ----------------------- | ------------------------------------------- |
| Country                 | Destination country name.                   |
| Tuition\_USD            | Annual average tuition fees (in USD).       |
| Living\_Cost\_USD       | Estimated yearly living costs (in USD).     |
| Visa\_Fee\_USD          | Standard student visa processing fee (USD). |
| Scholarships\_Available | Number of major scholarships available.     |
| Total\_Cost\_USD        | Sum of tuition, living, and visa fees.      |

---

## 🚀 Getting Started

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
   jupyter notebook notebooks/Cost_of_International_Education_Analysis.ipynb
   ```
3. **Explore Analysis**

   * Data cleaning and validation steps
   * Descriptive statistics for each cost component
   * Visualizations: bar charts, box plots, cost comparisons

---

## 📈 Key Insights

* **Tuition vs. Living Costs**: Some countries have low tuition but high living expenses, and vice versa.
* **Visa Fees Impact**: Visa processing fees are a small fraction but vary significantly.
* **Scholarship Availability**: Correlates moderately with overall cost.

(See the notebook for detailed charts and commentary.)

---

## 🛠️ Contributing 

Feel free to submit:

* Bug reports via GitHub Issues
* New analyses or extended visualizations via Pull Requests

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## 📄 License 

Licensed under the [MIT License](LICENSE).
