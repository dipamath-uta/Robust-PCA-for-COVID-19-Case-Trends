import pandas as pd

# Load your latest cleaned file
df = pd.read_csv("C:/Users/dipac/Downloads/covid-vax-project/clean_weekly_filtered2.csv", encoding="cp1252")

# Ensure numeric values
df['new_cases'] = pd.to_numeric(df['new_cases'], errors='coerce')
df['population'] = pd.to_numeric(df['population'], errors='coerce')

# Compute new cases per 100k people
df['new_cases_per_100k'] = (df['new_cases'] / df['population']) * 100000

# Optional: check quick summary
print(df[['country', 'continent', 'new_cases', 'population', 'new_cases_per_100k']].head())

# Save new dataset with the added column
output_path = "C:/Users/dipac/Downloads/covid-vax-project/clean_weekly_with_100k.csv"
df.to_csv(output_path, index=False)

print(f"✅ File saved successfully at:\n{output_path}")
