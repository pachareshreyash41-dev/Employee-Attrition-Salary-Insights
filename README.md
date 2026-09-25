# Employee Attrition & Salary Insights

An end-to-end data analysis and machine learning project aimed at identifying key drivers of employee turnover, analyzing compensation distributions, and evaluating predictive models for employee attrition.

---

## 📌 Project Overview

Employee turnover can significantly impact an organization's performance, culture, and bottom line. This project simulates a realistic employee dataset containing 500 records and applies statistical modeling (Logistic Regression) alongside machine learning algorithms (Random Forest) to uncover what causes employees to stay or leave.

Key goals of this analysis:
* Measure how factors such as **Overtime Hours**, **Years at Company**, **Monthly Salary**, and **Job Satisfaction** influence attrition.
* Build interpretable statistical models to understand log-odds impact ($p$-values, coefficients).
* Evaluate machine learning classification performance using precision, recall, and ROC-AUC metrics.

---

## 🛠️ Tech Stack & Libraries

* **Language:** Python 3.x
* **Environment:** Jupyter Notebook / JupyterLab
* **Data Manipulation:** `pandas`, `numpy`
* **Statistical Modeling:** `statsmodels`
* **Machine Learning:** `scikit-learn` (`RandomForestClassifier`, `train_test_split`, metrics)
* **Data Visualization:** `matplotlib`, `seaborn`

---

## 📊 Dataset Overview

The dataset contains **500 synthetic employee records** generated using a realistic logit probability model.

| Feature | Type | Description |
| :--- | :--- | :--- |
| `EmployeeID` | String | Unique identifier for each employee (`EMP-0001` to `EMP-0500`) |
| `Department` | Categorical | Department (`Sales`, `R&D`, `Human Resources`) |
| `YearsAtCompany` | Integer | Total years spent at the company (1 to 14 years) |
| `MonthlySalary` | Float | Monthly salary capped between $3,000 and $15,000 |
| `OvertimeHours` | Integer | Monthly overtime hours worked (0 to 39 hours) |
| `JobSatisfaction` | Categorical | Level of job satisfaction (`Low`, `Medium`, `High`, `Very High`) |
| `Attrition` | Binary | Target variable (`Yes` if left, `No` if stayed) |

---

## 📈 Key Analysis & Insights

1. **Job Satisfaction vs. Attrition Rate:**
   * **Low Satisfaction:** ~48.00% attrition rate.
   * **Medium Satisfaction:** ~30.97% attrition rate.
   * **High Satisfaction:** ~17.83% attrition rate.
   * **Very High Satisfaction:** ~7.95% attrition rate.

2. **Logistic Regression Findings:**
   * **Overtime Hours ($p < 0.001$):** Strong positive contributor to employee turnover.
   * **Job Satisfaction - Low ($p < 0.001$):** Significantly increases the likelihood of attrition.
   * **Years at Company ($p < 0.001$):** Has a protective effect; employees with longer tenure are less likely to leave.

3. **Random Forest Classification Performance:**
   * **ROC-AUC Score:** ~0.655
   * **Overall Accuracy:** ~71%

---

## 🚀 How to Run

1. **Clone the repository or download the project files:**
   ```bash
   git clone [https://github.com/your-username/employee-attrition-insights.git](https://github.com/your-username/employee-attrition-insights.git)
   cd employee-attrition-insights
