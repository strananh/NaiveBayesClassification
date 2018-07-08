* The Naive Bayes classification is run through command line, such as:
	> py .\PracticalProject-NaiveBayesClassification.py
	read data ...
	build dictionary ...
	transform training dataset to doc-term matrix ...
	use Mutual Information to select k-best features ...
	fit Multinomial Naive Bayes ...
	predict test dataset
				 precision    recall  f1-score   support

			  0       0.98      0.90      0.94      5000
			  1       0.91      0.98      0.94      5000

	avg / total       0.94      0.94      0.94     10000
	
* If there is no file 'dictionary', the new one is build. Otherwise, the dictionary is read from the file.
	