import numpy as np
from pyspark.sql.types import DoubleType
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from apk import * 
from util import *
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.linalg import SparseVector, DenseVector
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.evaluation import RankingMetrics

sc, sqlContext = init_spark(show_progress=False)
sc.addPyFile('util.py') 
sc.addPyFile('apk.py') 

#TODO:  concatenate valid_transformed.  
#Thought: If we randomly select a holdout set, then our estimate of error can be highly biased, even with large dataset. 
#Specially in case of unbalanced dataset
train = sqlContext.read.parquet('train_transformed')
train = train.withColumn("label", train["label"].cast(DoubleType()))
test= sqlContext.read.parquet('test_transformed')
test= test.withColumn("label", train["label"].cast(DoubleType()))
lr = LogisticRegression()


#Building the grid 
grid = ParamGridBuilder() \
	.addGrid(lr.maxIter, [0, 1, 5, 10, 15, 20, 25, 30]) \
	.addGrid(lr.regParam, [0.001, 0.01, 0.1, 1, 10, 100, 1000]) \
	.addGrid(lr.regType, ['l1', 'l2']) \
	.addGrid(r.elasticNetParam, [0.001, 0.01, 0.1, 1, 10, 100, 1000]) \
	.build()

metrics = RankingMetrics()
evluator =metrics.precisionAt(12)
cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator, numFolds=5)
cvModel = cv.fit(train)

#Testing overfitting/underfitting  
train_score = evaluator.evaluate(cvModel.transform(train))
test_score =  evaluator.evaluate(cvModel.transform(test))

#Saveing the model
bestModel = cvModel.bestModel
coefficients = DenseVector(bestModel.coefficients.toArray())
os.system('mkdir lr')
os.chdir(os.getcwd() + '/lr')
np.savez('coefficients', np.array(coefficients))  
np.save('intercept',bestModel.intercept)
np.save('train_score', train_score)
np.save('test_score', test_score)
