"""
Predicting bike demand:
 - preprocessing data using sklearn pipeline
 - applying linear regression
 - feature selection using statsmodel (P-value)
 - applying recursive feature selection
 - applying random forest
 - applying xgboost
 - selecting the best model
 - apllying the best model on test data
"""
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from statsmodels.api import OLS, add_constant
import xgboost as xgb
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sn
plt.style.use("seaborn")

# importing model packages

# xgboost

##read in data
bikes = pd.read_csv("train.csv", header=0,
                    index_col="datetime", parse_dates=True)
X = bikes.drop("count", axis=1)
y = bikes["count"]


# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43)
X_train.shape, y_train.shape

# check for empty values
X_train.isna().any()


# we need to transform certain columns:
# - create a column with hours
# - cretae a column with months
# - create a column with peek_hours
# - bin weather conditions in 2 groupes ("good","bad")
# - one_hot encode weather conditions
# - bin atemp in 3 groupes
# - one hot encode atemp
# - create interaction term between workday and hour
# - create a polynomail of degree 2 for humidity and atemp
# - min max scale of features
# - scale : atemp, atemp^2, humdity, humidity^2 , windspeed

####PREPROCESSING DATA######
def create_col_hour(df):
    hour = df.hour
    # month=df.dt.month
    return hour


def create_col_month(df):
    # hour=df.dt.hour
    month = df.month
    return month


X_train["hour"] = create_col_hour(X_train.index)
X_train["month"] = create_col_month(X_train.index)

# create peek hours
X_train["peek_hours"] = [1 if x in [7, 8, 9, 17, 18, 19]
                         else 0 for x in X_train["hour"]]

# make pipeline for weather conditions
pip_weather = make_pipeline(
    KBinsDiscretizer(n_bins=2, encode="onehot-dense", strategy="uniform")
)

# create pipeline for scaling and making polynomials
pip_scale_poly = make_pipeline(
    PolynomialFeatures(include_bias=False, degree=2),
    MinMaxScaler()
)


# implement column transformer

X_train.reset_index(inplace=True)

my_trans = ColumnTransformer(
    [
        ("bin_weather", pip_weather, ["weather"]),
        ("bin and encode atemp", KBinsDiscretizer(n_bins=3,
                                                  encode="onehot-dense", strategy="uniform"), ["atemp"]),
        ("interaction term work and hour", PolynomialFeatures(
            interaction_only=True, include_bias=False, degree=2), ["workingday", "hour"]),
        ("poly 2nd degree and scale", pip_scale_poly, ["atemp", "humidity"]),
        ("scale", MinMaxScaler(), ["windspeed"]),
        ("passthrough", "passthrough", ["datetime", "peek_hours"]),
        ("one hot encode", OneHotEncoder(), ["month", "season"]),
    ]
)

my_trans.fit(X_train)
X_trans = my_trans.transform(X_train)
X_trans = pd.DataFrame(X_trans, columns=["good_weath_cond", "bad_weath_cond", "low_temp",
                                         "medium_temp", "high_temp", "workingday", "hour", "inter_work_hour",
                                         "atemp", "humidity", "atemp^2", "interaction_atemp_hum", "humidity^2", "windspeed", "datetime",
                                         "peek_hours", "Jan", "Feb", "March", "Apr", "May", "Jun", "july", "Aug", "Sept", "Oct", "Nov", "Dec", "Spring", "Summer",
                                         "Autumn", "Winter"])

X_trans.set_index("datetime", inplace=True)
X_trans.head()

# transforming columns to numeric
X_trans = X_trans.apply(pd.to_numeric, errors="ignore")

# transform y
y_train_trans = np.log(y_train)

# for correlation purpose only
X_corr = pd.concat([X_trans, y_train_trans], axis=1)

sn.heatmap(abs(X_corr.corr()[["count"]]),
           cmap="RdBu_r", center=0.0, annot=True)
sn.heatmap(abs(X_trans.corr()), cmap="RdBu_r", center=0.0)

# inspect correlation amongst features
mask = np.triu(np.ones_like(X_corr.corr(), dtype=np.bool))
sn.heatmap(abs(X_corr.corr()), mask=mask)
# based on correlation map we can exclude:
#- workingday
# inteaction term btwe atemp and humidity
# medium_temp for correlation amongs (high_temp and low_temp)
# good_weather_cond for correlation with bad_weather_cond

features_sub = ["bad_weath_cond", "low_temp",
                "high_temp", "hour", "inter_work_hour",
                "atemp", "humidity", "atemp^2", "humidity^2", "windspeed", "month",
                "peek_hours", "season", "count"]  # count added for corelatin purpose only

sn.heatmap(abs(X_corr[features_sub].corr()[["count"]]),
           cmap="RdBu_r", center=0.0, annot=True)

# define RMSLE function


def RMSLe_(y_act, y_train):
    return np.sqrt(mean_squared_log_error(y_act, y_train))


###LINEAR RGRESSION#####

linear_regression = LinearRegression()
linear_regression.fit(X_trans, y_train_trans)

linear_regression.score(X_trans, y_train_trans)
y_pred_full = linear_regression.predict(X_trans)


# CV for lin reg
cross_val_lin = cross_val_score(
    linear_regression, X_trans, y_train_trans, cv=5)
rmsle_log_reg = RMSLe_(y_train_trans, y_pred_full)

output = pd.DataFrame([["linear reg", cross_val_lin.mean(), cross_val_lin.std(), rmsle_log_reg]],
                      columns=["model", "R2 mean", "R2 std", "RMSLE"])

# lasso rgerssion

lasso_m_ = Lasso()
alpha = [0.001, 0.005, 0.01, 0.3, 0.1, 0.3, 0.5, 0.7, 1]
score_lasso = []
for a in alpha:
    lasso_m = Lasso(max_iter=500, alpha=a)
    lasso_m.fit(X_trans, y_train_trans)
    score1 = lasso_m.score(X_trans, y_train_trans)
    score_lasso.append(score1)

lasso_best = Lasso(max_iter=500, alpha=0.001)
lasso_best.fit(X_trans, y_train_trans)
crossval_lass = cross_val_score(lasso_best, X_trans, y_train_trans, cv=5)
pred_lasso = lasso_best.predict(X_trans)
rmsle_lasso = RMSLe_(y_train_trans, pred_lasso)


output = output.append(pd.DataFrame([["linear reg lasso", crossval_lass.mean(), crossval_lass.std(), rmsle_lasso]],
                                    columns=["model", "R2 mean", "R2 std", "RMSLE"]))

#### FEATURE SELECTION####

X_train_sm = add_constant(X_trans)
X_train_sm.head()

# fitting model on full data
# - singluar matrix --> t_values are not reliable
# would eliminate variables:
## - Autumn
## - Jan
## - Feb, March, Apr, JUn, Aug,Oct
# - summer, spring, November
lin_reg_sm = OLS(endog=y_train_trans, exog=X_train_sm)
result = lin_reg_sm.fit()
print(result.summary())


subset_2 = ["good_weath_cond", "low_temp",
            "medium_temp", "high_temp", "workingday", "hour", "inter_work_hour",
            "atemp", "humidity", "atemp^2", "interaction_atemp_hum", "humidity^2",
            "peek_hours", "Spring",
            "Winter"]
lin_reg_ = OLS(endog=y_train_trans, exog=X_trans[subset_2])
result1 = lin_reg_.fit()
print(result1.summary())
# let"s fit a model on the selected sub features

# regression on a subset of features
linear_regression2 = LinearRegression()
linear_regression2.fit(X_trans[subset_2], y_train_trans)
linear_regression2.score(X_trans[subset_2], y_train_trans)
y_pred = linear_regression2.predict(X_trans[subset_2])
rmlse_lin_2 = np.sqrt(mean_squared_log_error(y_train_trans, y_pred))

cross_val_sub1 = cross_val_score(
    linear_regression2, X_trans, y_train_trans, cv=5)


output = output.append(
    {"model": "lin based in P-value (15)", "MSE mean": cross_val_sub1.mean(), "MSE std": cross_val_sub1.std(),
     "RMSLE": rmlse_lin_2}, ignore_index=True
)


# recurssive feature selection with cross validation
# linear regression is not really getting better with recurssive feature elimination
# we do not seem to have so much linear dependency amongst varaible and regressors
# we will continue with the selected features
estimator = Lasso(max_iter=500, alpha=0.001)
selector = RFE(estimator, n_features_to_select=15, step=1)
selector = selector.fit(X_trans, y_train_trans)

cross_val_sub1 = cross_val_score(selector, X_trans, y_train_trans, cv=5)

cross_val_sub1.mean(), cross_val_sub1.std()
y_pred_1 = selector.predict(X_trans)
rmlse_lin_3 = np.sqrt(mean_squared_log_error(y_train_trans, y_pred_1))

output = output.append(
    {"model": "lin based on RFE (15)", "MSE mean": cross_val_sub1.mean(), "MSE std": cross_val_sub1.std(),
     "RMSLE": rmlse_lin_3}, ignore_index=True
)


####END Feature selection for linear model###

# random forest for fullmodel
random_forest_base = RandomForestRegressor(n_estimators=100, n_jobs=-1, max_depth=5,
                                           oob_score=True)
random_forest_base.fit(X_trans, y_train_trans)
random_forest_base.score(X_trans, y_train_trans)
pred_rand = random_forest_base.predict(X_trans)
rmsle_rand = RMSLe_(y_train_trans, pred_rand)
cross_val = cross_val_score(random_forest_base, X_trans, y_train_trans, cv=5)

# check the feature importance
feat_importance = pd.DataFrame(
    random_forest_base.feature_importances_, index=X_trans.columns, columns=["var"])
feat_importance.sort_values(by="var", ascending=True).plot.barh()
plt.title("Random forest feature importance")

# appending RF resuts
output = output.append(
    {"model": "RF all", "R2 mean": cross_val.mean(), "R2 std": cross_val.std(),
     "RMSLE": rmsle_rand}, ignore_index=True
)

# optimize through CV search the selected subfeatures

param_grid = {
    'max_depth': [3, 5, 6, 7, 8],
}

grid_search = GridSearchCV(
    random_forest_base, param_grid, cv=ShuffleSplit(n_splits=10))
my_search = grid_search.fit(X_trans, y_train_trans)

ranked_res1 = pd.DataFrame(
    my_search.cv_results_).sort_values('rank_test_score')

rf_grid = my_search.predict(X_trans)
rf_grid_rmsle = RMSLe_(y_train_trans, rf_grid)

output = output.append(
    {"model": "RF grid search (max_deth 8)", "R2 mean": ranked_res1["mean_test_score"][4], "R2 std": ranked_res1["std_test_score"][4],
     "RMSLE": rf_grid_rmsle}, ignore_index=True
)

# Gradient Boosting

xgb = XGBRegressor(n_estimators=50, max_depth=5,
                   learning_rate=0.1, random_state=42)

xgb.fit(X_trans, y_train_trans)

cross_val_xgb = cross_val_score(xgb, X_trans, y_train_trans, cv=5)
pred_xgb = xgb.predict(X_trans)
rmlse_xgb = RMSLe_(y_train_trans, pred_xgb)

output = output.append(
    {"model": "GB 0.1 ", "R2 mean": cross_val_xgb.mean(), "R2 std": cross_val_xgb.std(),
     "RMSLE": rmlse_xgb}, ignore_index=True
)


# grid search for GB
param_grid_xgb = {
    'learning_rate': [0.1, 0.01, 0.2, 0.3, 0.5],
    'n_estimators': [10, 50, 100]
}

grid_search = GridSearchCV(xgb, param_grid_xgb, cv=ShuffleSplit(n_splits=10))
my_search_xgb = grid_search.fit(X_trans, y_train_trans)

ranked_res_xgb = pd.DataFrame(
    my_search_xgb.cv_results_).sort_values('rank_test_score')

my_search_xgb.fit(X_trans, y_train_trans)
pred_xgb_grid = xgb.predict(X_trans)
rmlse_xgb_grid = RMSLe_(y_train_trans, pred_xgb_grid)

# printng feature importance
pd.DataFrame(my_search_xgb.best_estimator_.feature_importances_, index=X_trans.columns,
             columns=["var"]).sort_values(by="var", ascending=True).plot.barh()
plt.title("Optimized XGB feature importance")

output = output.append(
    {"model": "GB LR Grid 0.3; n_est100 ", "R2 mean": 0.922743,
     "R2 std": 0.005188,
     "RMSLE": rmlse_xgb_grid}, ignore_index=True
)


#######PLOTTING RESULT on TRAIN DATA####

y_lin_reg = pd.concat([pd.Series(np.exp(y_pred_full)),
                       pd.Series(y_train.index)], axis=1)
y_lin_reg.set_index("datetime", inplace=True)


y_lasso = pd.concat([pd.Series(np.exp(pred_lasso)),
                     pd.Series(y_train.index)], axis=1)
y_lasso.set_index("datetime", inplace=True)

y_randrf = pd.concat([pd.Series(np.exp(pred_rand)),
                      pd.Series(y_train.index)], axis=1)
y_randrf.set_index("datetime", inplace=True)

rf_grid = pd.concat([pd.Series(np.exp(rf_grid)),
                     pd.Series(y_train.index)], axis=1)
rf_grid.set_index("datetime", inplace=True)

y_xgb = pd.concat([pd.Series(np.exp(pred_xgb_grid)),
                   pd.Series(y_train.index)], axis=1)
y_xgb.set_index("datetime", inplace=True)


# plot weekly data
plt.plot(y_train.resample("W").sum())
plt.plot(rf_grid.resample("W").sum())
plt.plot(y_xgb.resample("W").sum())
plt.plot(y_lasso.resample("W").sum())
plt.legend(["actual", "Rf", "XGB", "Lasso"])
plt.title("Train predictions vs actual predictions (daily)")

# plot monthly data
plt.plot(y_train.resample("M").sum())
plt.plot(rf_grid.resample("M").sum())
plt.plot(y_xgb.resample("M").sum())
plt.plot(y_lasso.resample("M").sum())
plt.legend(["actual", "Rf", "XGB", "Lasso"])
plt.title("Train predictions vs actual predictions (monthly)")

# plot monthly data
plt.plot(y_train.resample("M").sum())
plt.plot(rf_grid.resample("M").sum())
plt.plot(y_xgb.resample("M").sum())
plt.legend(["actual", "Rf", "XGB", "Lasso"])
plt.title("Train predictions vs actual predictions (monthly) only tree models")


# TEST DATA

X_test["hour"] = create_col_hour(X_test.index)
X_test["month"] = create_col_month(X_test.index)
X_test["peek_hours"] = [1 if x in [7, 8, 9, 17, 18, 19]
                        else 0 for x in X_test["hour"]]

X_test.reset_index(inplace=True)
my_trans.fit(X_test)
X_trans_test = my_trans.transform(X_test)
X_trans_test = pd.DataFrame(X_trans_test, columns=["good_weath_cond", "bad_weath_cond", "low_temp",
                                                   "medium_temp", "high_temp", "workingday", "hour", "inter_work_hour",
                                                   "atemp", "humidity", "atemp^2", "interaction_atemp_hum", "humidity^2", "windspeed", "datetime",
                                                   "peek_hours", "Jan", "Feb", "March", "Apr", "May", "Jun", "july", "Aug", "Sept", "Oct", "Nov", "Dec", "Spring", "Summer",
                                                   "Autumn", "Winter"])

X_trans_test.set_index("datetime", inplace=True)
X_trans_test.head()

# transforming columns to numeric
X_trans_test = X_trans_test.apply(pd.to_numeric, errors="ignore")

# transform the y columns
y_test_trans = np.log(y_test)

# test the models

my_search_xgb.score(X_trans_test, y_test_trans)
pred_XGV = my_search_xgb.predict(X_trans_test)
np.sqrt(mean_squared_log_error(y_test_trans, pred_XGV))


random_forest_base.score(X_trans_test, y_test_trans)
pred_rand = random_forest_base.predict(X_trans_test)
np.sqrt(mean_squared_log_error(y_test_trans, pred_rand))

# plot

rf_grid_test = pd.concat(
    [pd.Series(np.exp(pred_XGV)), pd.Series(y_test.index)], axis=1)
rf_grid_test.set_index("datetime", inplace=True)

y_xgb_test = pd.concat([pd.Series(np.exp(pred_rand)),
                        pd.Series(y_test.index)], axis=1)
y_xgb_test.set_index("datetime", inplace=True)

# plot weekly data
plt.plot(y_train.resample("W").sum())
# plt.plot(y_lin_reg.resample("w").sum())
# plt.plot(y_lasso.resample("W").sum())
plt.plot(rf_grid.resample("W").sum())
plt.plot(y_xgb.resample("W").sum())
plt.legend(["actual", "Rf", "XGB"])
plt.title("Random Forest and XGB vs actual (weekly) on test")

# plot monthly data
plt.plot(y_train.resample("M").sum())
# plt.plot(y_lin_reg.resample("w").sum())
# plt.plot(y_lasso.resample("W").sum())
plt.plot(rf_grid.resample("M").sum())
plt.plot(y_xgb.resample("M").sum())
plt.legend(["actual", "Rf", "XGB"])
plt.title("Random Forest and XGB vs actual (monthly) on test")


# Actual test data on Kaggel
bikes_test = pd.read_csv("test.csv", header=0,
                         index_col="datetime", parse_dates=True)
bikes_test["hour"] = create_col_hour(bikes_test.index)
bikes_test["month"] = create_col_month(bikes_test.index)
bikes_test["peek_hours"] = [1 if x in [7, 8, 9, 17, 18, 19]
                            else 0 for x in bikes_test["hour"]]

bikes_test.reset_index(inplace=True)
my_trans.fit(bikes_test)
bikes_test = my_trans.transform(bikes_test)
bikes_test = pd.DataFrame(bikes_test, columns=["good_weath_cond", "bad_weath_cond", "low_temp",
                                               "medium_temp", "high_temp", "workingday", "hour", "inter_work_hour",
                                               "atemp", "humidity", "atemp^2", "interaction_atemp_hum", "humidity^2", "windspeed", "datetime",
                                               "peek_hours", "Jan", "Feb", "March", "Apr", "May", "Jun", "july", "Aug", "Sept", "Oct", "Nov", "Dec", "Spring", "Summer",
                                               "Autumn", "Winter"])

bikes_test.set_index("datetime", inplace=True)
bikes_test.head()

# transforming columns to numeric
bikes_test = bikes_test.apply(pd.to_numeric, errors="ignore")

# predict XGB
pred_XGV_test = my_search_xgb.predict(bikes_test)

y_xgb_pred = pd.concat([pd.Series(np.exp(pred_XGV_test)),
                        pd.Series(bikes_test.index)], axis=1)

y_xgb_pred.columns = ["count", "datetime"]
y_xgb_pred.to_csv(
    r'C:/Users/Daniela/Documents/Spiced/Predicting_bike_demand/prediction_bike_demand_D.csv', index=False, header=True)
