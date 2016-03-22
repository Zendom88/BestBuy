from constants import new_train_file
from util import readfile
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

if __name__ == "__main__":
	reader = readfile(new_train_file)
	
	# makin a corpus from training data
	corpus = list()
	info = list()
	for idx,data in enumerate(reader):
		corpus.append(data[3])
		# user | sku | cat | time
		info.append([data[0],data[1],data[2], data[4]])
		if idx > 5000: break
	#end for
	info = pd.DataFrame(info, columns=["user","sku","cat","click_time"], dtype="category")
	
	# make a vector of count. removing features that only appears 1 time only
	CV = CountVectorizer(min_df = 1, stop_words="english", ngram_range=(1,2))
	feature_data = CV.fit_transform(corpus)
	feature_data = feature_data.toarray()
	feature_names = CV.get_feature_names()
	features_df = pd.DataFrame(feature_data,columns=feature_names)
	
	# make click_time data as features. Assume that click_time is a categorical type
	corpus = [time for time in info.click_time]
	CV = CountVectorizer(min_df = 0, stop_words=None)
	time_data = CV.fit_transform(corpus)
	time_features = pd.DataFrame(time_data.toarray(), columns=CV.get_feature_names())
	features_df = pd.concat([time_features,features_df], axis=1)
	
	# getting hot category
	# make hot_category as a training data
	hot_category_list = info.cat.value_counts()
	hot_category_list = hot_category_list[hot_category_list > hot_category_list.quantile(0.95)]
	train_hot_info = info[info.cat.isin(hot_category_list.index)]
	train_hot_features = features_df.loc[train_hot_info.index,:]
	train_hot = pd.concat([train_hot_info,train_hot_features], axis=1, join_axes=[train_hot_info.index])
	
	# non hot_category list as training data
	non_hot_category_list = info.cat.value_counts()
	non_hot_category_list = non_hot_category_list[non_hot_category_list <= non_hot_category_list.quantile(0.95)]
	train_non_hot_info = info[info.cat.isin(non_hot_category_list.index)]
	train_non_hot_features = features_df.loc[train_non_hot_info.index, :]
	train_non_hot = pd.concat([train_non_hot_info,train_non_hot_features], axis=1, join_axes=[train_non_hot_info.index])	
	
	# make a new field called hot_cat
	# this will make the prediction if the test data is in hot_category or not
	# user | sku | cat | hot_cat | w1 | w2 | wn...
	info.loc[train_hot_info.index,"hot_cat"] = pd.Series(1,index=info.index)
	info.loc[train_non_hot_info.index,"hot_cat"] = pd.Series(0,index=info.index)
	
	train_data = pd.concat([info,features_df], axis=1, join_axes=[info.index])
	column_headers = train_data.columns # make header before converting from pandas to numpy array
	
	# user | sku | cat | hot_cat | w1 w2 w3 ... wn
	train_data = np.array(train_data)
	
	# split to train and test
	train, test = train_test_split(train_data,test_size=0.2)
	train = pd.DataFrame(train,columns=column_headers)
	test = pd.DataFrame(test,columns=column_headers)
	
	# preparing training data to predict hot_cat
	# p(hot_cat|query)
	x_train = np.array(train.iloc[:,5:])
	y_train = np.array(train.hot_cat).astype(int)
	x_test = np.array(test.iloc[:,5:])
	y_test = np.array(test.hot_cat).astype(int)
	
	# this classifer will tell you whether this is hot category or not
	mnb_hot_or_not = MultinomialNB(alpha=0.1)
	mnb_hot_or_not.fit(x_train,y_train)
	y_pred = mnb_hot_or_not.predict(x_test)
	print("p(hot_cat|query) = {}".format(accuracy_score(y_test,y_pred)))
	test["pred_hot_cat"] = y_pred
	
	# preparing the data in order to find hot category
	# only predict data with pred_hot_cat == 1
	# assume this is my test data
	test_hot_features = test[test.pred_hot_cat == 1].iloc[:,5:test.shape[1]-1]
	test_hot_info = test[test.pred_hot_cat == 1].iloc[:,:5]
	test_hot = pd.concat([test_hot_info,test_hot_features], axis=1, join_axes=[test_hot_info.index])
	
	test_non_hot_features = test[test.pred_hot_cat == 0].iloc[:,5:test.shape[1]-1]
	test_non_hot_info = test[test.pred_hot_cat == 0].iloc[:,:5]
	test_non_hot = pd.concat([test_non_hot_info,test_non_hot_features], axis=1, join_axes=[test_non_hot_info.index])
		
	# this classifier will tell you what is the category if it is hot category
	#p(cat|query) in hot category
	x_train_hot = np.array(train_hot_features)
	y_train_hot = np.array(train_hot_info.cat)
	x_test_hot = np.array(test_hot_features)
	y_test_hot = np.array(test_hot_info.cat)
	
	mnb_predict_cat_for_hot = MultinomialNB(alpha=0.1)
	mnb_predict_cat_for_hot.fit(x_train_hot,y_train_hot)
	y_hot_pred = mnb_predict_cat_for_hot.predict(x_test_hot)
	print("p(cat|query) in hot_cat: {}".format(accuracy_score(y_test_hot,y_hot_pred)))
	test_hot_info.loc[:,"pred_cat"] = y_hot_pred
	
	# in otherwise, this classifier will classify what is the category for non_hot_category
	# p(cat|query) in non_hot category
	x_train_non_hot = np.array(train_non_hot_features)
	y_train_non_hot = np.array(train_non_hot_info.cat)
	x_test_non_hot = np.array(test_non_hot_features)
	y_test_non_hot = np.array(test_non_hot_info.cat)

	mnb_predict_cat_for_hot = MultinomialNB(alpha=0.1)
	mnb_predict_cat_for_hot.fit(x_train_non_hot,y_train_non_hot)
	y_non_hot_pred = mnb_predict_cat_for_hot.predict(x_test_non_hot)
	print("p(cat|query) in non_hot_cat: {}".format(accuracy_score(y_test_non_hot,y_non_hot_pred)))
	test_non_hot_info.loc[:,"pred_cat"] = y_non_hot_pred

	# using this p(sku|cat)
	# makin corpus of category. Assume that category is a string. So we can use CountVectorizer.
	corpus = [cat for cat in train_hot_info.cat]
	CV = CountVectorizer(min_df = 0, stop_words = None)
	x_train = CV.fit_transform(corpus)
	x_train = pd.DataFrame(x_train.toarray(),columns=CV.get_feature_names())
	y_train = np.array(train_hot_info.sku)
	train_features = CV.get_feature_names()
	
	corpus = [cat for cat in test_hot_info.pred_cat]
	x_test = CV.fit_transform(corpus)
	x_test = pd.DataFrame(x_test.toarray(), columns=CV.get_feature_names())
	y_test = np.array(test_hot_info.sku)
	
	if len(x_train.columns) >= len(x_test.columns):
		for feature in x_train.columns:
			if feature not in x_test: x_test[feature] = 0
	else:
		for feature in x_test.columns:
			if feature not in x_train: x_train[feature] = 0
	#end if
	
	x_train = np.array(x_train)
	y_train = np.array(y_train)
	x_test = np.array(x_test)
	y_test = np.array(y_test)
	mnb_sku_given_cat = MultinomialNB(alpha=0.1)
	mnb_sku_given_cat.fit(x_train, y_train)
	y_pred = mnb_sku_given_cat.predict(x_test)
	print("p(sku|cat) in hot_cat: {}".format(accuracy_score(y_test,y_pred)))
	
	corpus = [cat for cat in train_non_hot_info.cat]
	CV = CountVectorizer(min_df = 0, stop_words = None)
	x_train = CV.fit_transform(corpus)
	x_train = pd.DataFrame(x_train.toarray(),columns=CV.get_feature_names())
	y_train = train_non_hot_info.sku
	train_features = CV.get_feature_names()
	
	corpus = [cat for cat in test_non_hot_info.pred_cat]
	x_test = CV.fit_transform(corpus)
	x_test = pd.DataFrame(x_test.toarray(), columns=CV.get_feature_names())
	y_test = test_non_hot_info.sku
		
	if len(x_train.columns) >= len(x_test.columns):
		for feature in x_train.columns:
			if feature not in x_test: x_test[feature] = 0
	else:
		for feature in x_test.columns:
			if feature not in x_train: x_train[feature] = 0
	#end if
	
	x_train = np.array(x_train)
	y_train = np.array(y_train)
	x_test = np.array(x_test)
	y_test = np.array(y_test)
	mnb_sku_given_cat = MultinomialNB(alpha=0.1)
	mnb_sku_given_cat.fit(x_train, y_train)
	y_pred = mnb_sku_given_cat.predict(x_test)
	print("p(sku|cat) in non_hot_cat: {}".format(accuracy_score(y_test,y_pred)))
	
	# another method to train the data
	# i pick the category that already predicted from nb
	# and train the data
	for pred_cat in test_hot_info.pred_cat.value_counts().index:
		x_train = train_hot_info[train_hot_info.cat.isin([pred_cat])]
		x_train = train_hot_features.loc[x_train.index,:]
		y_train = train_hot_info.loc[x_train.index,:].sku
		
		x_test = test_hot_info[test_hot_info.pred_cat.isin([pred_cat])]
		x_test = test_hot_features.loc[x_test.index,:]
		tmp_index = x_test.index
		y_test = test_hot_info.loc[x_test.index,:].sku
		
		x_train = np.array(x_train)
		y_train = np.array(y_train)
		x_test = np.array(x_test)
		y_test = np.array(y_test)
		mnb_sku_given_query = MultinomialNB(alpha=0.1)
		mnb_sku_given_query.fit(x_train,y_train)
		y_pred = mnb_sku_given_query.predict(x_test)
		test_hot_info.loc[tmp_index,"pred_sku"] = y_pred
	#end for
	true_value = np.array(test_hot_info.sku,dtype="<U7")
	pred_value = np.array(test_hot_info.pred_sku,dtype="<U7")
	print("p(sku|query) for each cat in hot: {}".format(accuracy_score(true_value,pred_value)))
	
	for pred_cat in test_non_hot_info.pred_cat.value_counts().index:
		x_train = train_non_hot_info[train_non_hot_info.cat.isin([pred_cat])]
		x_train = train_non_hot_features.loc[x_train.index,:]
		y_train = train_non_hot_info.loc[x_train.index,:].sku
		
		x_test = test_non_hot_info[test_non_hot_info.pred_cat.isin([pred_cat])]
		x_test = test_non_hot_features.loc[x_test.index,:]
		tmp_index = x_test.index
		y_test = test_non_hot_info.loc[x_test.index,:].sku

		x_train = np.array(x_train)
		y_train = np.array(y_train)
		x_test = np.array(x_test)
		y_test = np.array(y_test)
		mnb_sku_given_query = MultinomialNB(alpha=0.1)
		mnb_sku_given_query.fit(x_train,y_train)
		y_pred = mnb_sku_given_query.predict(x_test)
		test_non_hot_info.loc[tmp_index,"pred_sku"] = y_pred
	#end for
	true_value = np.array(test_non_hot_info.sku,dtype="<U7")
	pred_value = np.array(test_non_hot_info.pred_sku,dtype="<U7")
	print("p(sku|query) for each cat in non_hot: {}".format(accuracy_score(true_value,pred_value)))
		