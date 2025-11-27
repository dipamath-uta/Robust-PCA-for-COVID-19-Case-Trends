import os
import pandas as pd

BASE = r"C:/Users/dipac/Downloads/covid-vax-project"

# Input file (your uploaded file)
INPUT = os.path.join(BASE, "clean_weekly_with_100k.csv")

# ----------------- LOAD DATA -----------------
df = pd.read_csv(INPUT, parse_dates=["date"])
df = df.sort_values("date")

# Ensure numeric
df["new_cases_per_100k"] = pd.to_numeric(df["new_cases_per_100k"], errors="coerce")
df["pfv_per_hundred"]    = pd.to_numeric(df["pfv_per_hundred"], errors="coerce")
df["population"]         = pd.to_numeric(df["population"], errors="coerce")

# ----------------- WEEKLY MEAN -----------------
df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)

weekly_mean = (
    df.groupby(["continent", "who_region", "country", "week"], as_index=False)
      .agg({
          "new_cases_per_100k": "mean",
          "pfv_per_hundred": "mean",
          "population": "mean"
      })
)

OUT = os.path.join(BASE, "clean_weekly_MEAN_latest.csv")
weekly_mean.to_csv(OUT, index=False)

print("Saved weekly MEAN dataset:", OUT)
