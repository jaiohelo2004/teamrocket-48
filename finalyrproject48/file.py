import mysql.connector
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="JaI#2004",
    database="moodledb"
)

print("Connected to MySQL")

query = "SELECT * FROM mdl_course"

df = pd.read_sql(query, conn)



print("Columns in dataset:")
print(df.columns)


print(df.head())
print("Dataset shape:", df.shape)



print("\nMissing values per column:")
print(df.isnull().sum())


#remove null values

threshold = len(df) * 0.5

df_clean = df.dropna(axis=1, thresh=threshold)

print("\nColumns remaining after removing high-missing columns:")
print(df_clean.columns)



# Detect constant columns
constant_columns = []

for col in df_clean.columns:
    if df_clean[col].nunique() <= 1:
        constant_columns.append(col)

print("\nConstant columns detected:")
print(constant_columns)

# Remove constant columns
df_clean2 = df_clean.drop(columns=constant_columns)

print("\nColumns remaining after removing constant columns:")
print(df_clean2.columns)


# Select only numeric columns
numeric_df = df_clean2.select_dtypes(include=['int64','float64'])

# Create correlation matrix
corr_matrix = numeric_df.corr()

print("\nCorrelation Matrix:")
print(corr_matrix)

# Detect highly correlated columns
correlated_columns = set()

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.9:
            colname = corr_matrix.columns[i]
            correlated_columns.add(colname)

print("\nHighly correlated columns:")
print(correlated_columns)

# Remove them
df_final = df_clean2.drop(columns=correlated_columns)

print("\nFinal columns after correlation filtering:")
print(df_final.columns)


#correlation heatmap

plt.figure(figsize=(10,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")


# plt.title("Correlation Matrix Heatmap")
#plt.show()

print("\nColumns removed due to high correlation:")
print(correlated_columns)



#ranking the attributes
# convert non-numeric columns
df_model = df_final.copy()
df_model = pd.get_dummies(df_model, drop_first=True)

# choose a target column (example: category)
target = "id"

X = df_model.drop(columns=[target])
y = df_model[target]

model = RandomForestRegressor()
model.fit(X, y)

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Column": X.columns,
    "Importance": importance
})

importance_df = importance_df.sort_values(by="Importance", ascending=False)

print("\nColumn Importance Ranking:")
print(importance_df)

df_final.to_csv("optimized_dataset.csv", index=False)

print("CSV created successfully")
