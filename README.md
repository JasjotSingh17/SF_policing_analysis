# SF-Policing-Analysis

# 🚔 SF Police Stop Patterns — Statistical Analysis & Data Modelling

A data analytics project investigating racial disparities, time-based patterns, and demographic differences in San Francisco police stop data — using hypothesis testing and interactive Power BI dashboards built on a purpose-designed star schema.

---

## 🚀 Key Takeaways

- Discovered that **search rates are not independent of driver race** — confirmed through a chi-square test (p < 0.001), with Black and Hispanic drivers searched at significantly higher rates than the overall average
- Found that **night stops (21:00–05:59) produce a significantly higher arrest rate than day stops** — confirmed through a two-proportion z-test (p < 0.001)
- Identified **statistically significant differences in subject age across police districts** — confirmed through one-way ANOVA with Tukey HSD post-hoc testing
- Designed a **star schema data model with 4 dimension tables and 1 fact table** across 905,070 stop records, powering interactive Power BI dashboards with live cross-filtering

---

## 📊 Dashboard Preview

### Stop Overview, Trends & Outcomes by Race
<img width="1304" height="1252" alt="image" src="https://github.com/user-attachments/assets/d07d7ca0-fb6d-4e4f-925a-f15884b09b87" />


<img width="2086" height="962" alt="image" src="https://github.com/user-attachments/assets/299b3502-73c4-496f-bed7-51c5654ca515" />

<img width="1350" height="750" alt="image" src="https://github.com/user-attachments/assets/bc33ceeb-3635-4a3a-814c-3557dcfbdc55" />


---

## 🔬 Hypothesis Test Evidence

### Test 1: Search Rate by Driver Race (Chi-Square)

<img width="1350" height="750" alt="image" src="https://github.com/user-attachments/assets/9e561892-2615-466e-affb-fc3499947e31" />


### Test 2: Night vs. Day Arrest Rate (Z-Test)
<img width="1350" height="750" alt="image" src="https://github.com/user-attachments/assets/0424741c-8ffd-4141-9afa-8a37a019f97a" />



---

## 📌 Problem

SF police stop records contain rich information about who gets stopped, searched, and arrested — but raw counts don't tell the whole story. This project investigates three specific questions:

> *Is search rate statistically independent of driver race, or is there a measurable disparity?*
> *Are night-time stops genuinely more likely to result in arrest, or is the difference random chance?*
> *Does the average age of stopped subjects differ meaningfully across police districts?*

Using 905,070 stop records from the Stanford Open Policing Project (2007–2016), this analysis applies rigorous hypothesis testing to distinguish real patterns from noise, and presents the findings through a structured Power BI dashboard built on a proper star schema.

---

## 🧠 Approach

### 1. Data Cleaning & Feature Engineering

- Loaded and filtered 905,070 raw stop records, retaining 13 relevant columns
- Parsed date strings into year, month, quarter, day-of-week, and hour features for time analysis
- Bucketed subject age into demographic groups and extracted readable time-of-day periods
- Standardised boolean columns (TRUE/FALSE strings → Python booleans) and flagged implausible age values

### 2. Data Modelling — Star Schema

Structured the cleaned data into a **star schema** to support efficient Power BI reporting:

| Table | Description | Rows |
| --- | --- | --- |
| `fact_stops` | One row per police stop — foreign keys + measures | 905,070 |
| `dim_date` | Calendar attributes (year, month, quarter, is_weekend) | ~3,000 |
| `dim_driver` | Stopped person's demographics (race, sex, age group) | ~60 |
| `dim_location` | District and stop type | ~30 |
| `dim_outcome` | Outcome flags (arrested, cited, searched, warned) | ~20 |

The four dimension tables link to `fact_stops` via surrogate integer keys, enabling Power BI's cross-filtering to work across all dimensions simultaneously.

### 3. Statistical Analysis & Hypothesis Testing

Conducted four hypothesis tests, each chosen for a specific statistical reason:

| Test | Method | Why This Method |
| --- | --- | --- |
| Is search rate independent of race? | Chi-Square Test of Independence | Two categorical variables (race, searched yes/no) |
| Is night arrest rate higher than day? | Two-Proportion Z-Test | Comparing proportions between two independent groups |
| Does contraband hit rate vary by search basis? | Chi-Square Test of Independence | Two categorical variables (search basis, contraband found) |
| Does subject age differ across districts? | One-Way ANOVA + Tukey HSD | Comparing means across 3+ groups with post-hoc pair identification |

For each test, the script reports the test statistic, p-value, effect size (Cramér's V or 95% CI), and a plain-English conclusion against α = 0.05.

---

## 📈 Key Insights

- **Search rates vary significantly by race** — the chi-square result (p < 0.001) rejects independence; Cramér's V quantifies the effect size beyond statistical significance alone
- **Night stops are disproportionately arrest-heavy** — the z-test confirms this is not sampling noise; the 95% confidence interval on the rate difference excludes zero
- **Age distributions differ across districts** — ANOVA returns a significant F-statistic; Tukey HSD identifies which specific district pairs drive the difference
- **Stop volume has declined since 2009** — the year-on-year trend chart shows a consistent downward pattern across the data window

---

## ⚙️ Tech Stack

| Tool | Role |
| --- | --- |
| Python (Pandas, NumPy) | Data loading, cleaning, transformation, star schema construction |
| SciPy (stats) | Chi-square test, one-way ANOVA, F-test |
| Statsmodels | Two-proportion z-test, Tukey HSD post-hoc test |
| Matplotlib | Hypothesis test evidence charts exported as PNG |
| Power BI | Interactive dashboards built on star schema CSVs |

---

## ▶️ How to Run

```
# 1. Clone the repo
git clone https://github.com/JasjotSingh17/SF-Policing-Analysis.git
cd SF-Policing-Analysis

# 2. Install dependencies
pip install pandas numpy scipy statsmodels matplotlib seaborn

# 3. Download the dataset from the Stanford Open Policing Project
# https://openpolicing.stanford.edu/data/
# Select: San Francisco, CA → download the CSV

# 4. Place the CSV in the project folder and update FILE_PATH in the script
# Default expected filename: ca_san_francisco_2020_04_01.csv

# 5. Run the analysis
python sf_policing_analysis.py
```

> All output CSVs for Power BI are saved to `./powerbi_data/`. All hypothesis test charts are saved to `./charts/`. Import the five star schema CSVs into Power BI Desktop and connect the foreign keys in Model view to enable cross-filtering.

---

## 👤 Author

**Jasjot Singh**
University of Waterloo — BMath, Mathematical Studies (Minor: Computer Science)
[LinkedIn](https://linkedin.com/in/your-profile) | j367sing@uwaterloo.ca
