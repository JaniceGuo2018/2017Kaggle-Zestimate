# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 19:45:18 2017

@author: Administrator
"""

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

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

	# Normalization
	numerical_list = properties.columns[abs(properties.max())>1.0].drop('parcelid')
	norm_properties = properties[numerical_list]
	properties[numerical_list] = (norm_properties - norm_properties.mean())/(norm_properties.max() - norm_properties.min())

	return properties

train=pd.read_csv('C:/Users/Administrator/Desktop/kaggle/training_data/train_2016_v2.csv')
Cl_prop=pd.read_csv('C:/Users/Administrator/Desktop/kaggle/training_data/properties_2016.csv')
Cl_prop=cleanAndFillData(Cl_prop)
tr_df=pd.merge(train,Cl_prop,on='parcelid',how='left')
#print (tr_df.head())
x_col=[col for col in tr_df.columns if col not in['logerror'] if tr_df[col].dtype=='float64']

labels=[]
values=[]
for col in x_col:
    labels.append(col)
    values.append(np.corrcoef(tr_df[col].values,tr_df.logerror.values)[0,1])
corr_df=pd.DataFrame({'col_labels':labels,'corr_values':values})
corr_df=corr_df.sort_values(by='corr_values')

#ind=np.arange(len(labels))
#width=0.7
#fig, ax=plt.subplots(figsize=(12,40))
#rects=ax.barh(ind,np.array(corr_df.corr_values.values))
#ax.set_yticks(ind)
#ax.set_yticklabels(corr_df.col_labels.values,rotation='horizontal')
#ax.set_xlabel("Correlation coefficient")
#ax.set_title("correlation coefficient between different variables and logerror")
#plt.show()
##print(corr_df.head())
corr_df_non=pd.DataFrame(columns=['corr_labels','corr_values'])
corr_df_non=corr_df.ix[(corr_df['corr_values']>0.02)|(corr_df['corr_values']<-0.01)]



cols_used=corr_df_non.col_labels.tolist()
temp_df=tr_df[cols_used]
X,y=temp_df,tr_df.logerror
X_new=SelectKBest(f_regression,k=5).fit_transform(X,y)
#corrmap=temp_df.corr(method='spearman')
#fig, ax=plt.subplots(figsize=(8,8))                        
#sns.heatmap(corrmap,vmax=1.,square=True)
#plt.title("Selected 12 variables corrmap",fontsize=15)
#plt.show()


