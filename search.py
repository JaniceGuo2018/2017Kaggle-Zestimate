import pandas as pd
import numpy as np
import xgboost as xgb
import gc
import seaborn as sns
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV
import warnings

def cleanAndFillData(properties):
	# Drop the repeat features:
	# ------------------------------------------------------------------------
	# 'finishedsquarefeet15' <=> 'calculatedfinishedsquarefeet'
	# 'finishedsquarefeet50' <=> 'finishedfloor1squarefeet'
	# 'regionidcounty'       <=> 'filps'
	# 'taxdelinquencyflag'   <=> 'taxdelinquencyyear'
	# ------------------------------------------------------------------------
	properties.drop(['finishedsquarefeet15', 'finishedsquarefeet50', 'regionidcounty', 'taxdelinquencyflag'], \
					axis=1, inplace=True)

	# Drop some features (thought to be useless) temporary:
	# ------------------------------------------------------------------------
	# 'regionidneighborhood', 'propertyzoningdesc', 'regionidcity',
	# 'censustractandblock', 'propertycountylandusecode',
	# 'rawcensustractandblock', 'assessmentyear', 'finishedfloor1squarefeet',
	# 'finishedsquarefeet13', 'fireplaceflag', 'finishedsquarefeet12', 
	# 'regionidzip'
	# ------------------------------------------------------------------------
	properties.drop(['regionidneighborhood', 'propertyzoningdesc', 'regionidcity', 'censustractandblock', \
	 				 'propertycountylandusecode', 'rawcensustractandblock', 'assessmentyear', 'finishedfloor1squarefeet', \
	 				 'finishedsquarefeet13', 'fireplaceflag', 'finishedsquarefeet12', 'regionidzip']\
	 				 , axis=1, inplace=True)

	# Fill NaNs with zeros:
	# ------------------------------------------------------------------------
	# 'pooltypeid10', 'pooltypeid2', 'pooltypeid7', 'taxdelinquencyyear',
	# 'basementsqft', 'yardbuildingsqft26', 'finishedsquarefeet6', 
	# 'poolsizesum', 'yardbuildingsqft17', 'poolcnt', 'garagecarcnt',
	# 'garagetotalsqft', 'lotsizesquarefeet', 'fireplacecnt',
	# 'threequarterbathnbr'
	# ------------------------------------------------------------------------
	properties.pooltypeid10.fillna(value=0, inplace=True)
	properties.pooltypeid2.fillna(value=0, inplace=True)
	properties.pooltypeid7.fillna(value=0, inplace=True)
	properties.taxdelinquencyyear.fillna(value=0, inplace=True)
	properties.basementsqft.fillna(value=0, inplace=True)
	properties.yardbuildingsqft26.fillna(value=0, inplace=True)
	properties.finishedsquarefeet6.fillna(value=0, inplace=True)
	properties.poolsizesum.fillna(value=0, inplace=True)
	properties.yardbuildingsqft17.fillna(value=0, inplace=True)
	properties.poolcnt.fillna(value=0, inplace=True)
	properties.garagecarcnt.fillna(value=0, inplace=True)
	properties.garagetotalsqft.fillna(value=0, inplace=True)
	properties.lotsizesquarefeet.fillna(value=0, inplace=True)
	properties.fireplacecnt.fillna(value=0, inplace=True)
	properties.threequarterbathnbr.fillna(value=0, inplace=True)

	# Fill NaNs with ones:
	# ------------------------------------------------------------------------
	# 'numberofstories', 'unitcnt'
	#
	# ------------------------------------------------------------------------
	properties.numberofstories.fillna(value=1, inplace=True)
	properties.unitcnt.fillna(value=1, inplace=True)

	# Fill gaps forward
	# ------------------------------------------------------------------------
	# 'fips', 'propertylandusetypeid', 'roomcnt',
	# 'bathroomcnt', 'bedroomcnt', 'taxamount', 
	# 'latitude', 'longitude'
	# ------------------------------------------------------------------------
	properties.fips.fillna(method='pad', inplace=True)
	properties.propertylandusetypeid.fillna(method='pad', inplace=True)
	properties.roomcnt.fillna(method='pad', inplace=True)
	properties.bathroomcnt.fillna(method='pad', inplace=True)
	properties.bedroomcnt.fillna(method='pad', inplace=True)
	# properties.regionidzip.fillna(method='pad', inplace=True)
	properties.latitude.fillna(method='pad', inplace=True)
	properties.longitude.fillna(method='pad', inplace=True)

	# Fill NaNs with group's mean
	# ====================================================================
	#            features            :               group by
	# --------------------------------------------------------------------
	#          'yearbuilt'           :                'fips'
	#  'structuretaxvaluedollarcnt'  :                'fips'
	#          'taxamount'           :                'fips'
	#      'taxvaluedollarcnt'       :                'fips'
	#    'landtaxvaluedollarcnt'     :                'fips'
	#   'finishedfloor1squarefeet'   :                'fips'
	# 'calculatedfinishedsquarefeet' :                'fips'
	#         'fullbathcnt'          :                'fips'
	#      'calculatedbathnbr'       :                'fips'
	# ====================================================================
	properties['yearbuilt'] = 2017 - properties.groupby("fips").yearbuilt.transform(lambda x: x.fillna(x.mean()))
	properties['structuretaxvaluedollarcnt'] = properties.groupby("fips").structuretaxvaluedollarcnt.transform(lambda x: x.fillna(x.mean()))
	properties['taxamount'] = properties.groupby("fips").taxamount.transform(lambda x: x.fillna(x.mean()))
	properties['taxvaluedollarcnt'] = properties.groupby("fips").taxvaluedollarcnt.transform(lambda x: x.fillna(x.mean()))
	properties['landtaxvaluedollarcnt'] = properties.groupby("fips").landtaxvaluedollarcnt.transform(lambda x: x.fillna(x.mean()))
	properties['calculatedfinishedsquarefeet'] = properties.groupby("fips").calculatedfinishedsquarefeet.transform(lambda x: x.fillna(x.mean()))
	properties['fullbathcnt'] = properties.groupby("fips").fullbathcnt.transform(lambda x: x.fillna(x.mean()))
	properties['calculatedbathnbr'] = properties.groupby("fips").calculatedbathnbr.transform(lambda x: x.fillna(x.mean()))
	
	# Convert logical feature:
	# ------------------------------------------------------------------------
	# 'hashottuborspa'
	#
	# ------------------------------------------------------------------------
	properties['hashottuborspa'] = properties['hashottuborspa'].replace(True, 1)
	properties.hashottuborspa.fillna(value=0, inplace=True)

	# One-hot encoding (fill NaNs with one-hot-nans):
	# ------------------------------------------------------------------------
	# 'airconditioningtypeid', 'buildingclasstypeid', 'heatingorsystemtypeid',
	# 'propertylandusetypeid', 'storytypeid', 'typeconstructiontypeid'
	# 'architecturalstyletypeid', 'decktypeid', 'fips'
	# ------------------------------------------------------------------------
	airconditioning_dummies = pd.get_dummies(properties['airconditioningtypeid'], prefix='airconditioning', dummy_na=True)
	buildingclass_dummies = pd.get_dummies(properties['buildingclasstypeid'], prefix='buildingclasstype', dummy_na=True)
	heatingorsystem_dummies = pd.get_dummies(properties['heatingorsystemtypeid'], prefix='heatingorsystem', dummy_na=True)
	propertylanduse_dummies = pd.get_dummies(properties['propertylandusetypeid'], prefix='propertylanduse', dummy_na=True)
	story_dummies = pd.get_dummies(properties['storytypeid'], prefix='story', dummy_na=True)
	construction_dummies = pd.get_dummies(properties['typeconstructiontypeid'], prefix='typeconstruction', dummy_na=True)
	architecturalstyle_dummies = pd.get_dummies(properties['architecturalstyletypeid'], prefix='architecturalstyle', dummy_na=True)
	decktype_dummies = pd.get_dummies(properties['decktypeid'], prefix='decktype', dummy_na=True)
	buildingqualitytype_dummies = pd.get_dummies(properties['buildingqualitytypeid'], prefix='buildingqualitytype', dummy_na=True)
	fips_dummies = pd.get_dummies(properties['fips'], prefix='fips')

	properties.drop(['airconditioningtypeid', 'buildingclasstypeid', 'heatingorsystemtypeid', 'propertylandusetypeid',\
					 'storytypeid', 'typeconstructiontypeid', 'architecturalstyletypeid', 'decktypeid', 'buildingqualitytypeid', \
					 'fips'], axis=1, inplace=True)

	properties = properties.join(airconditioning_dummies)
	properties = properties.join(buildingclass_dummies)
	properties = properties.join(heatingorsystem_dummies)
	properties = properties.join(propertylanduse_dummies)
	properties = properties.join(story_dummies)
	properties = properties.join(construction_dummies)
	properties = properties.join(architecturalstyle_dummies)
	properties = properties.join(decktype_dummies)
	properties = properties.join(buildingqualitytype_dummies)
	properties = properties.join(fips_dummies)

	# Convert regionidzip code into latitude/longitude.


	# Normalization
	# numerical_list = properties.columns[abs(properties.max())>1.0].drop('parcelid')
	# norm_properties = properties[numerical_list]
	# properties[numerical_list] = (norm_properties - norm_properties.mean())/(norm_properties.max() - norm_properties.min())


	return properties

def mergeTrainData(train, properties):
	train = train.merge(properties, on='parcelid', how='left')
	# train = train.sample(frac=1).reset_index(drop=True)

	# train['transactiondate'] = pd.to_datetime(train['transactiondate'])
	# # train['transactionyear'] = train['transactiondate'].apply(lambda x: x.year)
	# train['transactionmonth'] = train['transactiondate'].apply(lambda x: x.month)
	# # train['transactionday'] = train['transactiondate'].apply(lambda x: x.day)

	# transactionmonth_dummies = pd.get_dummies(train['transactionmonth'], prefix='transactionmonth')
	# train.drop(['transactionmonth'], axis=1, inplace=True)
	# train = train.join(transactionmonth_dummies)
	# norm_train = train['transactionday']
	# # train['transactionday'] = (norm_train - norm_train.mean())/(norm_train.max() - norm_train.min())

	# Drop outliers
	train=train[ train.logerror > -0.4 ]
	train=train[ train.logerror < 0.42 ]

	return train

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

def main():
	sns.set(font_scale = 1.5)
	train = pd.read_csv('train.csv')
	x_train = train.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
	train['logerror'] = train.logerror.values.astype(np.float32)
	y_train = train['logerror']
	del train; gc.collect()
	# print(np.shape(x_train), np.shape(y_train))
	# print(type(x_train), type(y_train))

	# test = xgb.DMatrix('./buffer/test.buffer')

	# print('grid search by CV...')
	# cv_params = {'max_depth': [3, 5, 7], 
	# 			 'min_child_weight': [1, 3, 5]}
	# ind_params = {'learning_rate': 0.1,
	# 			  'n_estimators': 1000, 
	# 			  'subsample': 0.8, 
	# 			  'colsample_bytree': 0.8, 
	# 			  'objective': 'reg:linear'}
	# optimized_GBM = GridSearchCV(xgb.XGBRegressor(**ind_params),
	# 							 cv_params, 
	# 							 cv=5, 
	# 							 scoring=make_scorer(mean_squared_error))
	# optimized_GBM.fit(x_train, y_train)
	# print(optimized_GBM.grid_scores_)
	# print(optimized_GBM.best_scores_)
	# print(optimized_GBM.best_params_)

	# Best combination
	# max_depth = 5
	# min_child_weight = 3
	#
	# print('grid search by CV...')
	# cv_params = {'learning_rate': [0.1, 0.05, 0.01], 
	# 			 'subsample': [0.7, 0.8, 0.9]}
	# ind_params = {'max_depth': 5, 
	# 			  'n_estimators': 1000, 
	# 			  'min_child_weight': 3, 
	# 			  'colsample_bytree': 0.8, 
	# 			  'objective': 'reg:linear'}
	# optimized_GBM = GridSearchCV(xgb.XGBRegressor(**ind_params),
	# 							 cv_params, 
	# 							 cv=5, 
	# 							 scoring=make_scorer(mean_squared_error))
	# optimized_GBM.fit(x_train, y_train)
	# print(optimized_GBM.grid_scores_)
	# print(optimized_GBM.best_scores_)
	# print(optimized_GBM.best_params_)

	# Best combination
	# learning_rate = 0.01
	# subsample = 0.8
	# 
	# print('CV...')
	# train = pd.read_csv('train.csv')
	# x_train = train.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
	# train['logerror'] = train.logerror.values.astype(np.float32)
	# y_train = train['logerror']
	# dtrain = xgb.DMatrix(x_train, y_train)
	# del train, x_train, y_train; gc.collect()
	# cv_params = {'max_depth': 5, 
	# 			 'n_estimators': 1000, 
	# 			 'min_child_weight': 3, 
	# 			 'colsample_bytree': 0.8, 
	# 			 'learning_rate' : 0.01, 
	# 			 'subsample' : 0.8, 
	# 			 'objective': 'reg:linear',
	# 			 'verbose': 10, 
	# 			 'gamma': 0.01, 
	# 			 'silent': 1}
	# model = xgb.cv(params=cv_params, dtrain=dtrain, num_boost_round=3000,
	# 			   nfold=5, metrics=['rmse'], early_stopping_rounds=100)
	# print(model.tail(5))

	print('Training...')
	train = pd.read_csv('train.csv')
	x_train = train.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
	train['logerror'] = train.logerror.values.astype(np.float32)
	y_train = train['logerror']
	dtrain = xgb.DMatrix(x_train, y_train)
	del train, x_train, y_train; gc.collect()
	xgb_params = {'eta': 0.1, 
				  'max_depth': 5, 
				  'n_estimators': 1000, 
				  'min_child_weight': 3, 
				  'colsample_bytree': 0.8, 
				  'learning_rate' : 0.01, 
				  'subsample' : 0.8, 
				  'objective': 'reg:linear',
				  'verbose': 10, 
				  'gamma': 0.01, 
				  'silent': 1}
	model = xgb.train(params=xgb_params, dtrain=dtrain, num_boost_round=1849)
	print('train over, saving model...')
	model.save_model('0007.model')
	importances = model.get_fscore()
	print(importances)
	xgb.plot_importance(model)

	print('load sample-submission and merge data...')
	properties = pd.read_csv("./training_data/properties_2016.csv")
	properties = cleanAndFillData(properties)
	submission = pd.read_csv("./training_data/sample_submission.csv")
	submission.rename(columns={'ParcelId':'parcelid'},inplace=True)
	test = submission.merge(properties, on='parcelid', how='left')
	del submission; gc.collect()

	test.drop(['parcelid','201610', '201611', '201612', '201710', '201711', '201712'], axis=1, inplace=True)
	dtest = xgb.DMatrix(test)
	dtest.save_binary('test.buffer')
	del test; gc.collect()

	predictions = model.predict(dtest)
	submission = pd.read_csv("./training_data/sample_submission.csv")
	for item in submission.columns[submission.columns != 'ParcelId']:
		submission[item] = predictions
	print('writing predictions...')
	submission.to_csv("./training_data/sample_submission_v6.csv", index=False, float_format='%.4f')

if __name__ == '__main__':
	main()