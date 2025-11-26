import pandas as pd

# 🔹 Load the dataset first
merged_clean = pd.read_csv(
    r"C:\Users\dipac\Downloads\covid-vax-project\vax_cases_clean_minimal.csv"
)

# 🔹 Then save or manipulate it as needed
merged_clean.to_csv(
    r"C:\Users\dipac\Downloads\covid-vax-project\merged_clean_daily.csv",
    index=False
)

print("✅ File saved successfully!")
