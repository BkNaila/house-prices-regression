# %%
import pandas as pd
# %%
df = pd.read_csv('../data/train.csv')
# %%
df.shape
# %%
df.info()
# %%
df['SalePrice'].describe()
# %%
import matplotlib.pyplot as plt
df['SalePrice'].hist(bins=50)
plt.show()
# %%
df.isnull().sum().sort_values(ascending=False).head(20)
# %%
cols=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']
# %%
df[cols]=df[cols].fillna("None")
# %%
df.isnull().sum().sort_values(ascending=False).head(10)
# %%
df[['GarageYrBlt', 'MasVnrArea']] = df[['GarageYrBlt', 'MasVnrArea']].fillna(0)
# %%
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
# %%
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
# %%
df.isnull().sum().sum()
# %%
qual_map = {"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1, "None":0 }
qual_cols=["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
             "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]
# %%
df[qual_cols]=df[qual_cols].apply(lambda col : col.map(qual_map))
# %%
df[qual_cols].head()
# %%
df['BsmtExposure']=df['BsmtExposure'].map({'Gd':4, 'Av':3,'Mn':2, 'No':1, 'None':0})
# %%
df['GarageFinish']=df['GarageFinish'].map({'Fin':3,'RFn':2, 'Unf':1, 'None':0})

# %%
df['Functional']=df['Functional'].map({'Typ':7, 'Min1':6, 'Min2':5, 'Mod':4, 'Maj1':3, 'Maj2':2, 'Sev':1, 'Sal':0})

# %%
df['LotShape']=df['LotShape'].map({'Reg':3,'IR1':2, 'IR2':1, 'IR3':0})
# %%
df['PavedDrive']=df['PavedDrive'].map({'Y':2, 'P':1, 'N':0})
# %%
df['BsmtFinType1']=df['BsmtFinType1'].map({'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1, 'None':0})
# %%
df['BsmtFinType2']=df['BsmtFinType2'].map({'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1, 'None':0})
# %%
df['Street']=df['Street'].map({'Grvl':0, 'Pave':1})
# %%
df['CentralAir'] = df['CentralAir'].map({'N': 0, 'Y': 1})
# %%
cols_to_check = ['Neighborhood', 'MSZoning', 'HouseStyle', 'Exterior1st', 'Exterior2nd',
                'SaleType', 'SaleCondition', 'BldgType', 'RoofStyle', 'RoofMatl',
                'Foundation', 'Heating', 'GarageType', 'MiscFeature', 'Fence',
                'MasVnrType', 'Condition1', 'Condition2', 'LotConfig', 'LandContour',
                'LandSlope', 'Utilities']
for col in cols_to_check:
    print(cols,df[cols].nunique())
# %%
df =pd.get_dummies(df, columns=cols_to_check)
# %%
df.shape
# %%
df.select_dtypes(include='object').columns
# %%
df = pd.get_dummies(df, columns=['Alley', 'Electrical'])
# %%
df.select_dtypes(include='object').columns

# %%
import numpy as np
# %%
df['SalePrice']=np.log1p(df['SalePrice'])
# %%
X= df.drop(columns=['SalePrice', 'Id'])
y=df['SalePrice']
# %%
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val= train_test_split(X, y, test_size=0.2, random_state=42)
# %%
X_train.shape, X_val.shape
# %%
numeric_cols=['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
                'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
                '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageYrBlt', 'GarageArea',
                'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                'PoolArea', 'MiscVal']
# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[numeric_cols]=scaler.fit_transform(X_train[numeric_cols])
X_val[numeric_cols]=scaler.transform(X_val[numeric_cols])
# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
# %%
model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, preds))
r2 = r2_score(y_val, preds)

print("RMSE:", rmse)
print("R²:", r2)
# %%
preds_dollars = np.expm1(preds)
y_val_dollars = np.expm1(y_val)
rmse_dollars = np.sqrt(mean_squared_error(y_val_dollars, preds_dollars))
print("RMSE in $:", rmse_dollars)
# %%
from sklearn.linear_model import Ridge, Lasso
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_preds = ridge.predict(X_val)
print("Ridge RMSE:", np.sqrt(mean_squared_error(y_val, ridge_preds)))
print("Ridge R²:", r2_score(y_val, ridge_preds))
# %%
lasso = Lasso(alpha=0.001)
lasso.fit(X_train, y_train)
lasso_preds = lasso.predict(X_val)
print("Lasso RMSE:", np.sqrt(mean_squared_error(y_val, lasso_preds)))
print("Lasso R²:", r2_score(y_val, lasso_preds))
# %%
from sklearn.model_selection import GridSearchCV

ridge_params = {'alpha': [0.01, 0.1, 1, 10, 50, 100]}
ridge_grid = GridSearchCV(Ridge(), ridge_params, scoring='neg_root_mean_squared_error', cv=5)
ridge_grid.fit(X_train, y_train)
print("Best Ridge alpha:", ridge_grid.best_params_)
print("Best Ridge RMSE:", -ridge_grid.best_score_)

lasso_params = {'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01]}
lasso_grid = GridSearchCV(Lasso(max_iter=10000), lasso_params, scoring='neg_root_mean_squared_error', cv=5)
lasso_grid.fit(X_train, y_train)
print("Best Lasso alpha:", lasso_grid.best_params_)
print("Best Lasso RMSE:", -lasso_grid.best_score_)
# %%
best_ridge = Ridge(alpha=50)
best_ridge.fit(X_train, y_train)
best_preds = best_ridge.predict(X_val)
print("RMSE:", np.sqrt(mean_squared_error(y_val, best_preds)))
print("R²:", r2_score(y_val, best_preds))
# %%
for a in [1, 5, 10, 20, 50]:
    r = Ridge(alpha=a)
    r.fit(X_train, y_train)
    p = r.predict(X_val)
    print(a, np.sqrt(mean_squared_error(y_val, p)), r2_score(y_val, p))
# %%
for a in [0.01, 0.1, 0.5, 1, 2]:
    r = Ridge(alpha=a)
    r.fit(X_train, y_train)
    p = r.predict(X_val)
    print(a, np.sqrt(mean_squared_error(y_val, p)), r2_score(y_val, p))
# %%
final_model = Ridge(alpha=0.5)
final_model.fit(X_train, y_train)
final_preds = final_model.predict(X_val)

preds_dollars = np.expm1(final_preds)
y_val_dollars = np.expm1(y_val)
print("Final RMSE in $:", np.sqrt(mean_squared_error(y_val_dollars, preds_dollars)))
print("Final R²:", r2_score(y_val, final_preds))
# %%
