# %%
import pandas as pd
# %%
from pathlib import Path
path = Path('data/train.csv') if Path('data/train.csv').exists() else Path('../data/train.csv')
df = pd.read_csv(path)# %%
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
df['house_age'] = df['YrSold'] - df['YearBuilt']
df['remod_age'] = df['YrSold'] - df['YearRemodAdd']
# %%
df['remod_age'] = df['remod_age'].clip(lower=0)
# %%
df[['house_age', 'remod_age']].describe()
# %%
df['total_sqft']=df['1stFlrSF']+df['2ndFlrSF']+df['TotalBsmtSF']
df['total_bathrooms']=df['FullBath']+0.5*df['HalfBath']+df['BsmtFullBath']+0.5*df['BsmtHalfBath']
df['has_pool'] = (df['PoolArea'] > 0).astype(int)
df['has_garage'] = (df['GarageArea'] > 0).astype(int)
df['has_fireplace'] = (df['Fireplaces'] > 0).astype(int)
# %%
df[['house_age', 'remod_age', 'total_sqft', 'total_bathrooms', 'has_pool', 'has_garage', 'has_fireplace']].describe()
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
                'PoolArea', 'MiscVal', 'house_age', 'remod_age', 'total_sqft', 'total_bathrooms']
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
from sklearn.linear_model import ElasticNet
elastic = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)
elastic.fit(X_train, y_train)
elastic_preds = elastic.predict(X_val)
print("ElasticNet RMSE:", np.sqrt(mean_squared_error(y_val, elastic_preds)))
print("ElasticNet R²:", r2_score(y_val, elastic_preds))
# %%
for a in [0.0001, 0.0003, 0.0005, 0.001]:
    e = ElasticNet(alpha=a, l1_ratio=0.5, max_iter=10000)
    e.fit(X_train, y_train)
    p = e.predict(X_val)
    print(a, np.sqrt(mean_squared_error(y_val, p)), r2_score(y_val, p))
# %%
import matplotlib.pyplot as plt

best = ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=10000)
best.fit(X_train, y_train)
preds = best.predict(X_val)

actual = np.expm1(y_val)
predicted = np.expm1(preds)

plt.scatter(actual, predicted, alpha=0.5)
plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], color='red')
plt.xlabel("Actual price ($)")
plt.ylabel("Predicted price ($)")
plt.show()
# %%
coefs = pd.Series(best.coef_, index=X_train.columns)
print(coefs.abs().sort_values(ascending=False).head(10))
# %%
print(X_train[['RoofMatl_ClyTile', 'Condition2_PosN', 'MSZoning_C (all)', 'Exterior1st_BrkComm', 'MiscFeature_TenC']].sum())
# %%
binary = [c for c in X_train.columns if X_train[c].nunique() == 2]
rare = [c for c in binary if X_train[c].sum() < 10]
print(len(rare), "rare columns dropped")

X_train2 = X_train.drop(columns=rare)
X_val2 = X_val.drop(columns=rare)

m = ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=10000)
m.fit(X_train2, y_train)
p = m.predict(X_val2)
print("RMSE:", np.sqrt(mean_squared_error(y_val, p)))
print("R²:", r2_score(y_val, p))
# %%
test_path = Path('data/test.csv') if Path('data/test.csv').exists() else Path('../data/test.csv')
test = pd.read_csv(test_path)
print(test.shape)
missing = test.isnull().sum().sort_values(ascending=False)
print(missing[missing > 0])
# %%
test[cols] = test[cols].fillna("None")
zero_cols = ['GarageYrBlt', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
             'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']
test[zero_cols] = test[zero_cols].fillna(0)
test['LotFrontage'] = test['LotFrontage'].fillna(df['LotFrontage'].median())
for c in test.columns[test.isnull().any()]:
    test[c] = test[c].fillna(test[c].mode()[0])

print(test.isnull().sum().sum())
# %%
test[qual_cols] = test[qual_cols].apply(lambda c: c.map(qual_map))
test['BsmtExposure'] = test['BsmtExposure'].map({'Gd':4, 'Av':3, 'Mn':2, 'No':1, 'None':0})
test['GarageFinish'] = test['GarageFinish'].map({'Fin':3, 'RFn':2, 'Unf':1, 'None':0})
test['Functional'] = test['Functional'].map({'Typ':7, 'Min1':6, 'Min2':5, 'Mod':4, 'Maj1':3, 'Maj2':2, 'Sev':1, 'Sal':0})
test['LotShape'] = test['LotShape'].map({'Reg':3, 'IR1':2, 'IR2':1, 'IR3':0})
test['PavedDrive'] = test['PavedDrive'].map({'Y':2, 'P':1, 'N':0})
fin_map = {'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1, 'None':0}
test['BsmtFinType1'] = test['BsmtFinType1'].map(fin_map)
test['BsmtFinType2'] = test['BsmtFinType2'].map(fin_map)
test['Street'] = test['Street'].map({'Grvl': 0, 'Pave': 1})
test['CentralAir'] = test['CentralAir'].map({'N': 0, 'Y': 1})

test['house_age'] = test['YrSold'] - test['YearBuilt']
test['remod_age'] = (test['YrSold'] - test['YearRemodAdd']).clip(lower=0)
test['total_sqft'] = test['1stFlrSF'] + test['2ndFlrSF'] + test['TotalBsmtSF']
test['total_bathrooms'] = test['FullBath'] + 0.5*test['HalfBath'] + test['BsmtFullBath'] + 0.5*test['BsmtHalfBath']
test['has_pool'] = (test['PoolArea'] > 0).astype(int)
test['has_garage'] = (test['GarageArea'] > 0).astype(int)
test['has_fireplace'] = (test['Fireplaces'] > 0).astype(int)
test = pd.get_dummies(test, columns=cols_to_check + ['Alley', 'Electrical'])

print(test.shape)
print("In train, missing from test:", set(X_train.columns) - set(test.columns))
print("In test, not in train:", set(test.columns) - set(X_train.columns))
# %%
ids = test['Id']
test_X = test.reindex(columns=X_train.columns, fill_value=0)

test_X[numeric_cols] = scaler.transform(test_X[numeric_cols])

print(test_X.shape)
print(test_X.isnull().sum().sum())
# %%
print(X_train.shape[1] == test_X.shape[1])
# %%
import joblib

test_preds = np.expm1(best.predict(test_X))
print(test_preds[:5])
print(test_preds.min(), test_preds.max())

submission = pd.DataFrame({'Id': ids, 'SalePrice': test_preds})
submission.to_csv('submission.csv', index=False)

joblib.dump(best, '../models/elasticnet_house_prices.joblib')
# %%
print("Predictions above 755k:", (test_preds > 755000).sum())

i = test_preds.argmax()
print("Id:", ids.iloc[i], "predicted:", test_preds[i])
print(test.loc[i, ['GrLivArea', 'total_sqft', 'LotArea', 'OverallQual', 'YearBuilt']])

rare_flags = [c for c in X_train.columns if X_train[c].nunique() == 2 and X_train[c].sum() < 10]
print([c for c in rare_flags if test_X.iloc[i][c] == 1])
# %%
test_preds = np.clip(test_preds, 34900, 755000)

submission = pd.DataFrame({'Id': ids, 'SalePrice': test_preds})
submission.to_csv('submission.csv', index=False)

joblib.dump(best, '../models/elasticnet_house_prices.joblib')