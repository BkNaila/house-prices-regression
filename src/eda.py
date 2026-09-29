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
df[PoolQC, MiscFeature, Alley, Fence, FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, MasVnrType].fillna('None')