import pandas as pd
import numpy as np
import xgboost as xgb
import gc

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

def main():
    
    print('load train and properties datas...')
    train = pd.read_csv("./training_data/train_2016_v2.csv")
    properties = pd.read_csv("./training_data/properties_2016.csv")

    print('clean and merge train data...')
    properties = cleanAndFillData(properties)
    train = mergeTrainData(train, properties)
    # print(train.isnull().any())

    x_train = train.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
    y_train = train['logerror'].values.astype(np.float32)
    y_mean = np.mean(y_train)
    print('clean train and properties buffer...')
    del train, properties; gc.collect()

    # fold_size = 80000
    # x_train, y_train, x_valid, y_valid= x_train[:fold_size], y_train[:fold_size], x_train[fold_size:], y_train[fold_size:]
    dtrain = xgb.DMatrix(x_train, y_train)
    # dvalid = xgb.DMatrix(x_valid, y_valid)
    print('saving dtrain.buffer and valid.buffer')
    # dtrain.save_binary('train.buffer')
    # dvalid.save_binary('valid.buffer')
    print('clean buffer...')
    # del x_train, y_train, x_valid, y_valid; gc.collect()
    del x_train, y_train; gc.collect()

    # XGBoost training.
    # XGBoost settings.
    xgb_params = {'eat': 0.033,
                  'max_depth': 6,
                  'subsample': 0.80,
                  'objective': 'reg:linear',
                  'eval_metric': 'mae',
                  'base_score': y_mean,
                  'silent': 1 
                  }
    # watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    print('training...')
    cv_result = xgb.cv(xgb_params, dtrain, nfold=5, num_boost_round=3000, \
    				   early_stopping_rounds=10, verbose_eval=10)
    num_boost_rounds = len(cv_result)
    # model = xgb.train(xgb_params, dtrain, num_boost_rounds, watchlist,\
    # 				  early_stopping_rounds=10, verbose_eval=10)
    model = xgb.train(xgb_params, dtrain, num_boost_rounds)
    print('train over, saving model...')
    model.save_model('0005.model')
    print('clean buffer...')
    # del dtrain, dvalid; gc.collect()
    del dtrain; gc.collect()

    print('load sample-submission and merge data...')
    properties = pd.read_csv("./training_data/properties_2016.csv")
    properties = cleanAndFillData(properties)
    submission = pd.read_csv("./training_data/sample_submission.csv")
    submission.rename(columns={'ParcelId':'parcelid'},inplace=True)
    test = submission.merge(properties, on='parcelid', how='left')
    del submission; gc.collect()
    
    test.drop(['parcelid','201610', '201611', '201612', '201710', '201711', '201712'], axis=1, inplace=True)
    dtest = xgb.DMatrix(test)
    del test; gc.collect()

    predictions = model.predict(dtest)
    
    submission = pd.read_csv("./training_data/sample_submission.csv")
    for item in submission.columns[submission.columns != 'ParcelId']:
    	submission[item] = predictions
    print('writing predictions...')
    submission.to_csv("./training_data/sample_submission_v4.csv", index=False, float_format='%.4f')
    xgb.plot_importance(model)
	# print 'XGBoost predictions:'
	# print pd.DataFrame(dtest)

	# 10-fold cross validation.
	# fold_size = 9000

	# for i in range(1,10):
	# 	s = (i-1)*fold_size
	# 	e = i*fold_size
	# 	x_valid_fold = x_train[s:e]
	# 	y_valid_fold = y_train[s:e]
	# 	x_train_fold = x_train[:s].append(x_train[e:], ignore_index=True)
	# 	y_train_fold = y_train[:s] + y_train[e:]

if __name__ == '__main__':
	main()