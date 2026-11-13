
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

DATA_PATH = Path("data/telecom_customer_churn.csv")

df = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# 2. BASIC INFORMATION
# --------------------------------------------------

print("\n========== DATASET OVERVIEW ==========")

print("Dataset shape:", df.shape)
print("Total rows:", df.shape[0])
print("Total columns:", df.shape[1])

# --------------------------------------------------
# 3. COLUMN NAMES AND DATA TYPES
# --------------------------------------------------

print("\n========== COLUMN NAMES ==========")

for column in df.columns:
    print(column)

print("\n========== DATA TYPES ==========")
print(df.dtypes)

# --------------------------------------------------
# 4. MISSING VALUES
# --------------------------------------------------

print("\n========== MISSING VALUES ==========")

missing = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage": (
        df.isnull().sum() / len(df) * 100
    ).round(2)
})

print(missing[missing["Missing Count"] > 0])

# --------------------------------------------------
# 5. DUPLICATE RECORDS
# --------------------------------------------------

print("\n========== DUPLICATES ==========")

print("Duplicate rows:", df.duplicated().sum())

# --------------------------------------------------
# 6. TARGET VARIABLE
# --------------------------------------------------

print("\n========== TARGET VARIABLE ==========")

if "Churn" in df.columns:
    print(df["Churn"].value_counts(dropna=False))
    print("\nTarget percentages:")
    print(
        (df["Churn"].value_counts(normalize=True) * 100)
        .round(2)
    )
else:
    print("WARNING: Churn column not found!")

# --------------------------------------------------
# 7. SAMPLE RECORDS
# --------------------------------------------------

print("\n========== FIRST 5 RECORDS ==========")

print(df.head().to_string())

print("\n========== INSPECTION COMPLETED ==========")
