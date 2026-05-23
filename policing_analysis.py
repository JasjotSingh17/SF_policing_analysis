import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, norm
import statsmodels.stats.proportion as smprop
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

FILE_PATH = "ca_san_francisco_2020_04_01.csv"   
OUTPUT_DIR = "powerbi_data"
CHART_DIR  = "charts"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHART_DIR,  exist_ok=True)

# Colour palette for charts
PALETTE = {
    "blue":   "#2563EB",
    "red":    "#DC2626",
    "green":  "#16A34A",
    "orange": "#EA580C",
    "purple": "#7C3AED",
    "grey":   "#6B7280",
}


df = pd.read_csv(
    FILE_PATH,
    parse_dates=["date"],           
    low_memory=False                
)


KEEP_COLS = [
    "date", "time", "district",
    "subject_race", "subject_sex", "subject_age",
    "type",
    "arrest_made", "citation_issued", "warning_issued", "outcome",
    "search_conducted", "contraband_found",
    "reason_for_stop",
]
df = df[[c for c in KEEP_COLS if c in df.columns]].copy()


df["year"]        = df["date"].dt.year
df["month"]       = df["date"].dt.month
df["month_name"]  = df["date"].dt.strftime("%b")   
df["day_of_week"] = df["date"].dt.dayofweek        
df["dow_name"]    = df["date"].dt.strftime("%a")   
df["quarter"]     = df["date"].dt.quarter

df["hour"] = pd.to_numeric(df["time"].str[:2], errors="coerce")

def hour_to_period(h):
    if pd.isna(h):    return "Unknown"
    if h < 6:         return "Late Night (0–5)"
    elif h < 12:      return "Morning (6–11)"
    elif h < 17:      return "Afternoon (12–16)"
    elif h < 21:      return "Evening (17–20)"
    else:             return "Night (21–23)"

df["time_of_day"] = df["hour"].apply(hour_to_period)


for col in ["arrest_made", "citation_issued", "warning_issued",
            "search_conducted", "contraband_found"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.upper().map(
            {"TRUE": True, "FALSE": False, "1": True, "0": False}
        )

if "subject_age" in df.columns:
    df["subject_age"] = pd.to_numeric(df["subject_age"], errors="coerce")
    df.loc[(df["subject_age"] < 10) | (df["subject_age"] > 100), "subject_age"] = np.nan

    bins   = [0, 17, 24, 34, 44, 54, 64, 120]
    labels = ["<18", "18–24", "25–34", "35–44", "45–54", "55–64", "65+"]
    df["age_group"] = pd.cut(df["subject_age"], bins=bins, labels=labels)

if "subject_race" in df.columns:
    df["subject_race"] = df["subject_race"].str.lower().str.strip()
    race_counts = df["subject_race"].value_counts(normalize=True)
    rare_races  = race_counts[race_counts < 0.01].index
    df.loc[df["subject_race"].isin(rare_races), "subject_race"] = "other/unknown"

if "district" in df.columns:
    df["district"] = df["district"].str.strip().str.title()

def make_dim(df, cols, id_col_name):
    dim = df[cols].drop_duplicates().dropna(subset=cols[:1]).reset_index(drop=True)
    dim[id_col_name] = dim.index + 1
    return dim

date_cols = ["date", "year", "month", "month_name", "quarter",
             "day_of_week", "dow_name"]
dim_date = (
    df[date_cols]
    .drop_duplicates(subset=["date"])
    .dropna(subset=["date"])
    .reset_index(drop=True)
    .sort_values("date")
)
dim_date["date_id"] = dim_date.index + 1
dim_date["is_weekend"] = dim_date["day_of_week"].isin([5, 6])

driver_cols = ["subject_race", "subject_sex", "age_group"]
dim_driver = make_dim(df, driver_cols, "driver_id")

location_cols = ["district", "type"]
dim_location = make_dim(df, location_cols, "location_id")

outcome_cols = ["outcome", "arrest_made", "citation_issued",
                "warning_issued", "search_conducted"]
dim_outcome = make_dim(df, outcome_cols, "outcome_id")

fact = df.copy()

fact = fact.merge(dim_date[["date", "date_id"]], on="date", how="left")
fact = fact.merge(dim_driver[driver_cols + ["driver_id"]], on=driver_cols, how="left")
fact = fact.merge(dim_location[location_cols + ["location_id"]], on=location_cols, how="left")
fact = fact.merge(dim_outcome[outcome_cols + ["outcome_id"]], on=[c for c in outcome_cols if c in fact.columns], how="left")

measure_cols = ["contraband_found", "subject_age", "hour", "reason_for_stop",
                "time_of_day"]
fact_stops = fact[
    ["date_id", "driver_id", "location_id", "outcome_id"] +
    [c for c in measure_cols if c in fact.columns]
].copy()
fact_stops["stop_id"] = fact_stops.index + 1   

for name, table in [("dim_date",     dim_date),
                    ("dim_driver",   dim_driver),
                    ("dim_location", dim_location),
                    ("dim_outcome",  dim_outcome),
                    ("fact_stops",   fact_stops)]:
    path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    table.to_csv(path, index=False)

ALPHA = 0.05   

def print_result(test_name, stat, p_value, extra=""):
    sig = "REJECT H₀ (statistically significant)" if p_value < ALPHA \
          else "FAIL TO REJECT H₀ (not significant)"
    print(f"\n  {'─'*58}")
    print(f"  TEST: {test_name}")
    print(f"  {'─'*58}")
    print(f"  Test statistic : {stat:.4f}")
    print(f"  p-value        : {p_value:.6f}  (α = {ALPHA})")
    if extra:
        print(f"  {extra}")
    print(f"  Conclusion     : {sig}")

t1 = df.dropna(subset=["subject_race", "search_conducted"]).copy()
t1["searched_int"] = t1["search_conducted"].astype(bool).astype(int)

contingency = pd.crosstab(t1["subject_race"], t1["searched_int"])
print("\n  Contingency table (counts):")
print(contingency.to_string())

search_rates = (
    t1.groupby("subject_race")["searched_int"]
    .agg(["sum", "count"])
    .rename(columns={"sum": "searched", "count": "total"})
)
search_rates["search_rate_%"] = (search_rates["searched"] / search_rates["total"] * 100).round(2)
print("\n  Search rates by race:")
print(search_rates.to_string())

chi2_stat, p_chi2, dof, expected = chi2_contingency(contingency)
n = contingency.values.sum()
cramers_v = np.sqrt(chi2_stat / (n * (min(contingency.shape) - 1)))

print_result(
    "Chi-Square: Search Rate vs. Driver Race",
    chi2_stat,
    p_chi2,
    f"Degrees of freedom : {dof}\n  Cramér's V (effect size) : {cramers_v:.4f}"
    f"\n  Interpretation: Cramér's V {'< 0.1 (small effect)' if cramers_v < 0.1 else '0.1–0.3 (medium)' if cramers_v < 0.3 else '> 0.3 (large effect)'}"
)

t2 = df.dropna(subset=["hour", "arrest_made"]).copy()
t2["is_night"]    = t2["hour"].isin(list(range(21, 24)) + list(range(0, 6)))
t2["arrested"]    = t2["arrest_made"].astype(bool).astype(int)

night = t2[t2["is_night"] == True]["arrested"]
day   = t2[t2["is_night"] == False]["arrested"]

n_night, n_day   = len(night), len(day)
x_night, x_day   = night.sum(), day.sum()
p_night, p_day   = x_night / n_night, x_day / n_day

print(f"\n  Night stops (21:00–05:59) : {n_night:,}  |  arrests: {x_night:,}  |  rate: {p_night:.2%}")
print(f"  Day stops   (06:00–20:59) : {n_day:,}  |  arrests: {x_day:,}  |  rate: {p_day:.2%}")

z_stat, p_z = smprop.proportions_ztest(
    [x_night, x_day],
    [n_night, n_day],
    alternative="larger"      
)

diff = p_night - p_day
se   = np.sqrt(p_night*(1-p_night)/n_night + p_day*(1-p_day)/n_day)
ci_lo, ci_hi = diff - 1.96*se, diff + 1.96*se

print_result(
    "Two-Proportion Z-Test: Night vs. Day Arrest Rate",
    z_stat,
    p_z,
    f"Rate difference (night − day) : {diff:+.4f}\n"
    f"  95% CI on difference         : [{ci_lo:+.4f}, {ci_hi:+.4f}]"
)

if "search_basis" in df.columns:
    t3 = df[df["search_conducted"] == True].dropna(
        subset=["search_basis", "contraband_found"]
    ).copy()
    t3["found_int"] = t3["contraband_found"].astype(bool).astype(int)

    ct3 = pd.crosstab(t3["search_basis"], t3["found_int"])
    hit_rates = (
        t3.groupby("search_basis")["found_int"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "hits", "count": "searches"})
    )
    hit_rates["hit_rate_%"] = (hit_rates["hits"] / hit_rates["searches"] * 100).round(2)
    print("\n  Hit rates by search basis:")
    print(hit_rates.to_string())

    chi2_t3, p_t3, dof_t3, _ = chi2_contingency(ct3)
    n3 = ct3.values.sum()
    cv3 = np.sqrt(chi2_t3 / (n3 * (min(ct3.shape) - 1)))
    print_result(
        "Chi-Square: Contraband Hit Rate vs. Search Basis",
        chi2_t3,
        p_t3,
        f"Degrees of freedom : {dof_t3}\n  Cramér's V (effect size) : {cv3:.4f}"
    )
else:
    print("  (search_basis column not found – skipping Test 3)")

t4 = df.dropna(subset=["district", "subject_age"]).copy()

# Only keep districts with ≥30 stops (law of large numbers)
dist_counts = t4["district"].value_counts()
valid_dists  = dist_counts[dist_counts >= 30].index
t4 = t4[t4["district"].isin(valid_dists)]

age_by_district = [
    group["subject_age"].values
    for _, group in t4.groupby("district")
    if len(group) >= 30
]
district_means = t4.groupby("district")["subject_age"].mean().sort_values()
print("\n  Mean subject age by district:")
print(district_means.round(2).to_string())

f_stat, p_anova = stats.f_oneway(*age_by_district)
print_result(
    "One-Way ANOVA: Subject Age vs. District",
    f_stat,
    p_anova,
    "Note: Significant F only tells us *some* district differs.\n"
    "  A post-hoc Tukey HSD test (see below) identifies *which* pairs differ."
)

if p_anova < ALPHA:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    tukey = pairwise_tukeyhsd(t4["subject_age"], t4["district"], alpha=ALPHA)
    print("\n  Tukey HSD pairwise comparisons (significant pairs only):")
    tukey_df = pd.DataFrame(data=tukey._results_table.data[1:],
                             columns=tukey._results_table.data[0])
    tukey_df["reject"] = tukey_df["reject"].astype(str)
    sig_pairs = tukey_df[tukey_df["reject"] == "True"]
    if len(sig_pairs) > 0:
        print(sig_pairs[["group1","group2","meandiff","p-adj","reject"]].to_string(index=False))
    else:
        print("  No individual pairs differ significantly after correction.")


stops_by_year = df.groupby("year").size().reset_index(name="total_stops")
stops_by_year.to_csv(os.path.join(OUTPUT_DIR, "summary_stops_by_year.csv"), index=False)


race_summary = (
    df.dropna(subset=["subject_race"])
    .groupby("subject_race")
    .agg(
        total_stops      =("arrest_made",     "count"),
        arrests          =("arrest_made",     lambda x: x.astype(bool).sum()),
        citations        =("citation_issued", lambda x: x.astype(bool).sum()),
        searches         =("search_conducted",lambda x: x.astype(bool).sum()),
    )
    .reset_index()
)
race_summary["arrest_rate"] = (race_summary["arrests"]   / race_summary["total_stops"]).round(4)
race_summary["citation_rate"] = (race_summary["citations"] / race_summary["total_stops"]).round(4)
race_summary["search_rate"] = (race_summary["searches"]  / race_summary["total_stops"]).round(4)
race_summary.to_csv(os.path.join(OUTPUT_DIR, "summary_outcomes_by_race.csv"), index=False)


district_summary = (
    df.dropna(subset=["district"])
    .groupby("district")
    .agg(
        total_stops =("arrest_made", "count"),
        arrest_rate =("arrest_made", lambda x: x.astype(bool).mean()),
        search_rate =("search_conducted", lambda x: x.astype(bool).mean()),
        avg_age =("subject_age", "mean"),
    )
    .reset_index()
    .round(4)
)
district_summary.to_csv(os.path.join(OUTPUT_DIR, "summary_by_district.csv"), index=False)

hour_summary = (
    df.dropna(subset=["hour"])
    .groupby("hour")
    .agg(
        total_stops =("arrest_made","count"),
        arrest_rate =("arrest_made", lambda x: x.astype(bool).mean()),
    )
    .reset_index()
    .round(4)
)
hour_summary.to_csv(os.path.join(OUTPUT_DIR, "summary_by_hour.csv"), index=False)

search_rates.reset_index().to_csv(
    os.path.join(OUTPUT_DIR, "hypothesis_test1_search_by_race.csv"), index=False
)

t2[["is_night","arrested"]].groupby("is_night")["arrested"].agg(
    arrests="sum", total="count"
).assign(arrest_rate=lambda x: x["arrests"]/x["total"]).reset_index().to_csv(
    os.path.join(OUTPUT_DIR, "hypothesis_test2_night_vs_day.csv"), index=False
)


# ── Charts ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
race_order = search_rates.sort_values("search_rate_%")
bars = ax.barh(race_order.index, race_order["search_rate_%"],
               color=PALETTE["blue"], edgecolor="white")
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
ax.set_xlabel("Search Rate (%)")
ax.set_title("Search Rate by Driver Race – SF Police Stops (2007–2016)",
             fontsize=12, fontweight="bold", pad=12)
ax.axvline(race_order["search_rate_%"].mean(), color=PALETTE["red"],
           linestyle="--", label=f"Overall avg: {race_order['search_rate_%'].mean():.1f}%")
ax.legend(fontsize=9)
ax.set_xlim(0, race_order["search_rate_%"].max() * 1.25)
plt.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "chart1_search_rate_by_race.png"), dpi=150)
plt.close()

# Chart 2: Stops by year
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(stops_by_year["year"], stops_by_year["total_stops"],
       color=PALETTE["blue"], edgecolor="white")
ax.set_xlabel("Year")
ax.set_ylabel("Total Stops")
ax.set_title("Total Police Stops by Year – San Francisco (2007–2016)",
             fontsize=12, fontweight="bold", pad=12)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "chart2_stops_by_year.png"), dpi=150)
plt.close()

# Chart 3: Arrest rate by time of day
arrest_by_period = (
    df.dropna(subset=["time_of_day","arrest_made"])
    .groupby("time_of_day")["arrest_made"]
    .apply(lambda x: x.astype(bool).mean() * 100)
    .reset_index(name="arrest_rate_%")
)
period_order = ["Late Night (0–5)","Morning (6–11)","Afternoon (12–16)",
                "Evening (17–20)","Night (21–23)"]
arrest_by_period["time_of_day"] = pd.Categorical(
    arrest_by_period["time_of_day"], categories=period_order, ordered=True
)
arrest_by_period = arrest_by_period.sort_values("time_of_day")
fig, ax = plt.subplots(figsize=(9, 5))
colors = [PALETTE["red"] if "Night" in p or "Late" in p else PALETTE["blue"]
          for p in arrest_by_period["time_of_day"].astype(str)]
bars = ax.bar(arrest_by_period["time_of_day"].astype(str),
              arrest_by_period["arrest_rate_%"], color=colors, edgecolor="white")
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
ax.set_xlabel("Time of Day")
ax.set_ylabel("Arrest Rate (%)")
ax.set_title("Arrest Rate by Time of Day – SF Police Stops",
             fontsize=12, fontweight="bold", pad=12)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "chart3_arrest_rate_by_time.png"), dpi=150)
plt.close()

# Chart 4: District mean age (ANOVA visualisation)
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(district_means.index, district_means.values, color=PALETTE["purple"],
        edgecolor="white")
ax.axvline(district_means.mean(), color=PALETTE["orange"], linestyle="--",
           label=f"Overall mean: {district_means.mean():.1f}")
ax.set_xlabel("Mean Subject Age")
ax.set_title("Mean Subject Age by Police District – SF (ANOVA)",
             fontsize=12, fontweight="bold", pad=12)
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "chart4_age_by_district.png"), dpi=150)
plt.close()




