

#import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



#suspress warnings
import warnings
warnings.filterwarnings('ignore')

#1. Source the data. Create dataframes from SQL statements. Join the tables to enrich the records with additional columns.
##I connected to the SQLite database, read and extracted data from 3 tables. Then I merged these tables to create a single a DataFrame with all house data + crime + tax rate.

import sqlite3

conn = sqlite3.connect("/content/TechPro-DataScience-Midseason-Dataset.db")

df_houses= pd.read_sql_query("SELECT * FROM houses", conn)
df_cities= pd.read_sql_query("SELECT * FROM cities", conn).rename(columns={"index": "City"})
df_energy_classes= pd.read_sql_query("SELECT * FROM energy_classes", conn).rename(columns={"index": "Energy Class"})

conn.close()


# Merge houses with cities on 'City'
df = pd.merge(df_houses, df_cities, on="City", how="left")

# Merge the result with energy_classes on 'Energy Class'
df = pd.merge(df, df_energy_classes, on="Energy Class", how="left")

# Show the first few rows of the merged dataframe
df.head()

df = df.drop(columns=["Floor","uuid"])

# Show the first few rows of the merged dataframe
df.head(10)

#2. Pre-process the data. Handle missing values, cast string values to their proper types, encode categorical variables, normalize the columns, etc
## Explore the DataFrame, Handle missing values,search for duplicates,  cast string values to their proper types, encode categorical variables, search for outliers , normalize the columns (?)

df.shape

df.nunique()

df.info()

# View missing data

df.isna().sum()

#  Drop rows missing critical fields or info that aren't so important and Fill less critical ones

df = df.dropna(subset=["Area","Price"])
df["Bedrooms"]=df["Bedrooms"].fillna(df["Bedrooms"].median())
df["Bathrooms"]=df["Bathrooms"].fillna(df["Bathrooms"].median())
df["Crime"] = df["Crime"].fillna(df["Crime"].median())
df["Tax Rate"] = df["Tax Rate"].fillna(df["Tax Rate"].median())
df["Year Built"] = df["Year Built"].fillna(df["Year Built"].median())
df["City"] = df["City"].fillna(df["City"].mode()[0])
df["Energy Class"] = df["Energy Class"].fillna(df["Energy Class"].mode()[0])

df.info()

df.shape
df.isna().sum()

# how many duplicates rows appears
df.duplicated().sum()

df = df.drop_duplicates()

df.dtypes

# Ensure Categorical and Numerical  columns has the right type

df["City"] = df["City"].astype(str)
df["Energy Class"] = df["Energy Class"].astype(str)
df["Bedrooms"] = df["Bedrooms"].astype(int)
df["Bathrooms"] = df["Bathrooms"].astype(int)
df["Price"] = df["Price"].astype(int)
df["Area"] = df["Area"].astype(int)
df["Year Built"] = df["Year Built"].astype(int)
df.info()

df.head(10)

#encode categorical variables
#df["City"].value_counts()
#df["Energy Class"].value_counts()

def convert_city_to_int(City):
    if City == 'Athens':
        return 1
    elif City == 'Thessaloniki':
        return 2
    elif City == 'Crete':
        return 3
    elif City == 'Thebes':
        return 4
    elif City == 'Larissa':
        return 5
    elif City == 'Trikala':
        return 6
    elif City == 'Lamia':
        return 7
    else:
        return 0

df["City"] = df["City"].apply(convert_city_to_int)

def convert_energy_to_int(Energy_Class):
    if Energy_Class == 'A':
        return 1
    elif Energy_Class == 'B':
        return 2
    elif Energy_Class == 'C':
        return 3
    elif Energy_Class == 'D':
        return 4
    elif Energy_Class == 'E':
        return 5
    elif Energy_Class == 'F':
        return 6
    else:
        return 0

df["Energy Class"] = df["Energy Class"].apply(convert_energy_to_int)

df.head(10)

#check for outliers using plots
df["Price"].plot(kind="hist")

import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot for Area
plt.figure(figsize=(6, 4))
sns.boxplot(x=df["Area"])
plt.title("Boxplot of Area")
plt.show()

df.describe()

#3. Perform an analysis on the dataset showing how each of the 8 features affects the final price.

df.corr()

# Plot heatmap
# Full correlation heatmap focused on all features' relationship with Price
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Heatmap Among All Features Including Price")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix
cor = df.corr(numeric_only=True)

# Plot the heatmap of correlation with respect to Price
plt.figure(figsize=(10, 6))
sns.heatmap(cor[["Price"]].sort_values(by="Price", ascending=False), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Feature Correlation with Price")
plt.show()

# Show sorted correlations as a reference
cor["Price"].sort_values(ascending=False)

#4. Train a linear regression model to predict the price according to the rest of the features.

# Define features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)


# Predict on the test set
y_pred = model.predict(X_test)


# 5. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n  RMSE: €{rmse:,.2f}")
print(f"  R²: {r2:.4f}")

# Optional: Visualize Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='teal')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # ideal line
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.tight_layout()
plt.show()

#5. Evaluate the regression model using RMSE.

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
#rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#rmse_f = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)



print("-------")
print(f"📏 Mean Squared Error (MSE): {mse:,.2f}")
print(f"📉 Mean Absolute Error (MAE): {mae:,.2f}")
print(f"📐 Root Mean Squared Error (RMSE): {rmse:,.2f} €")
print(f"📈 R-squared (R²): {r2:.4f}")

#! test my model

# Predict using the trained model
y_pred = model.predict(X_test)

# Compare predicted vs actual
comparison_df = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})
print(comparison_df.head(10))

#6. Perform 10-fold cross-validation to the model.
###10-Fold Cross-Validation is a method to evaluate how well your model will generalize to unseen data.
###You get 10 evaluation scores (e.g., RMSE), which you can average for a more reliable estimate of your model's performance.

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error

#rmse_scorer = make_scorer(mean_squared_error)
def rmse(y_test, y_pred):
    #return mean_squared_error(y_test, y_pred, squared=False)
    return np.sqrt(np.mean((y_test - y_pred) ** 2))

rmse_scorer = make_scorer(rmse)

# Set up 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=12)
model = LinearRegression()

# Perform cross-validation and get RMSE scores
rmse_scores = cross_val_score(model, X, y, scoring=rmse_scorer, cv=kf)

# Report results
print("🔟 10-Fold Cross-Validation RMSE Scores (Euros):")
print(np.round(rmse_scores, 2))
print(f"\n📊 Average RMSE: {rmse_scores.mean():,.2f}€")

#This shows the model has good generalization and isn’t overfitting to specific parts of the dataset.

##Normalize or scale features, If You Want to Improve (later)

#7. Train a DecisionTree regression model to predict the price according to the rest of the features.

from sklearn.tree import DecisionTreeRegressor

# Train Decision Tree
dt_model = DecisionTreeRegressor( random_state=12)
dt_model.fit(X_train, y_train)

# Predict
y_pred_dt = dt_model.predict(X_test)

# Evaluate
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
r2_dt = r2_score(y_test, y_pred_dt)

print(f"\n Decision Tree RMSE: {rmse_dt:,.2f}€")
print(f" Decision Tree R²: {r2_dt:.4f}")

#8. Evaluate the regression model using RMSE and compare it with the linear regression

import matplotlib.pyplot as plt

# Data
models = ['Linear Regression', 'Decision Tree']
rmse = [10172.76, 15008.01]
r2 = [0.9025, 0.7877]

# Plot
plt.figure(figsize=(8, 5))

# RMSE
plt.bar(models, rmse, color='lightblue', label='RMSE (€)')


plt.ylabel("RMSE (€)")
plt.title("Model Comparison: RMSE")
plt.show()

# R²
plt.figure(figsize=(8, 5))
plt.bar(models, r2, color='orange', label='R² Score')


plt.ylabel("R² Score")
plt.ylim(0, 1)
plt.title("Model Comparison: R² Score")
plt.show()

"""
✅ Your Linear Regression Model Results
 RMSE: €10,172.76

→ On average, your model’s predictions are off by about 10K€.
That’s quite low for real estate pricing — very good!

📈 R²: 0.9025
→ Your model explains 90.25% of the variance in housing prices.
That’s a strong fit — especially for a simple linear model.

💬 What does this mean in context?
Your linear model performs even better than the Decision Tree, which had:

RMSE ≈ 15,008.01€

R² ≈ 0.79



✅ This might suggest:

Relationships in your dataset are fairly linear

There's not much overfitting, since the performance is solid on the test set

Simpler models like Linear Regression might be ideal for this data (at least at this stage)


------

Linear Regression performs better overall.

The Decision Tree model might be overfitting or not capturing general patterns as well.

Trees tend to fit noise unless tuned properly (e.g., with max_depth, min_samples_leaf).
"""

#9. Try out different settings for the decission tree (`max_depth`, `min_samples_split`) to reduce overfitting

from sklearn.tree import DecisionTreeRegressor

# Train Decision Tree
dt_model = DecisionTreeRegressor( random_state=12)
dt_model.fit(X_train, y_train)

# Predict
y_pred_dt = dt_model.predict(X_test)

# Evaluate
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
r2_dt = r2_score(y_test, y_pred_dt)

print(f"\n Decision Tree RMSE: €{rmse_dt:,.2f}")
print(f" Decision Tree R²: {r2_dt:.4f}")

# Train Decision Tree No2
dt_model = DecisionTreeRegressor(max_depth=10, min_samples_split=30, random_state=12)
dt_model.fit(X_train, y_train)

# Predict
y_pred_dt = dt_model.predict(X_test)

# Evaluate
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
r2_dt = r2_score(y_test, y_pred_dt)

print(f"\n Decision Tree RMSE: €{rmse_dt:,.2f}")
print(f" Decision Tree R²: {r2_dt:.4f}")

# Settings to test
max_depth_values = [3, 5, 10, 15, None]
min_samples_split_values = [5, 15, 35, 50]

# Store results
results = []

for max_depth in max_depth_values:
    for min_split in min_samples_split_values:
        model = DecisionTreeRegressor(max_depth=max_depth,
                                      min_samples_split=min_split,
                                      random_state=12)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            'max_depth': max_depth if max_depth is not None else "None",
            'min_samples_split': min_split,
            'RMSE': rmse,
            'R2': r2
        })

df_results = pd.DataFrame(results)

# Pivot the RMSE results to prepare for heatmap
pivot = df_results.pivot(index='max_depth', columns='min_samples_split', values='RMSE')

plt.figure(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues")
plt.title("RMSE for Different Decision Tree Settings")
plt.ylabel("max_depth")
plt.xlabel("min_samples_split")
plt.show()

# Pivot the R² results to prepare for heatmap
pivot_r2 = df_results.pivot(index='max_depth', columns='min_samples_split', values='R2')

plt.figure(figsize=(8, 6))
sns.heatmap(pivot_r2, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("R² Score for Different Decision Tree Settings")
plt.ylabel("max_depth")
plt.xlabel("min_samples_split")
plt.show()
