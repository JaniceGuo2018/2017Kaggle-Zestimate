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
    print('Loading train and properties data...')
    train = pd.read_csv("./training_data/train_2016_v2.csv")
    properties = pd.read_csv("./training_data/properties_2016.csv")
    
    print('Train data shape is: ', np.shape(train))
    print('Cleaning and merging train data...')
    
    properties = cleanAndFillData(properties)
    train = mergeTrainData(train, properties)

    print('Saving train data...')
    print('Train data reshape in: ', np.shape(train))
    
    train.to_csv('train.csv', index=False, header=True)
    x_train = train.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
    y_train = train['logerror'].values.astype(np.float32)
    x_train.to_csv('x_train.csv', index=False, header=True)
    y_train.to_csv('y_train.csv', index=False, header=True)

    print('Train data save over...')
    del train, properties, x_train, y_train; gc.collect()

    print('Loading sample-submission and merge data...')
    properties = pd.read_csv("./training_data/properties_2016.csv")
    properties = cleanAndFillData(properties)
    submission = pd.read_csv("./training_data/sample_submission.csv")
    submission.rename(columns={'ParcelId':'parcelid'},inplace=True)
    test = submission.merge(properties, on='parcelid', how='left')
    del submission; gc.collect()
    
    test.drop(['parcelid','201610', '201611', '201612', '201710', '201711', '201712'], axis=1, inplace=True)
    test.to_csv('./training_data/test.csv', index=False, header=True)
    del test; gc.collect()

if __name__ == '__main__':
	main()