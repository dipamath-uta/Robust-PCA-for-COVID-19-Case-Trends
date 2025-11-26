# Save the cleaned daily-level dataset
merged_clean.to_csv(
    r"C:\Users\dipac\Downloads\covid-vax-project\merged_clean_daily.csv",
    index=False
)
print("✅ Daily clean data saved successfully!")

# Save the weekly aggregated dataset
weekly_data.to_csv(
    r"C:\Users\dipac\Downloads\covid-vax-project\weekly_data.csv",
    index=False
)
print("✅ Weekly aggregated data saved successfully!")
