from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import numpy as np
from pyspark.sql.types import DoubleType
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from apk import * 
from util import *
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.linalg import SparseVector, DenseVector
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

sc, sqlContext = init_spark(show_progress=False)
sc.addPyFile('util.py') 
sc.addPyFile('apk.py') 

#TODO:  concatenate valid_transformed 
train = sqlContext.read.parquet('train_transformed')
train = train.withColumn("label", train["label"].cast(DoubleType()))
rf = RandomForestClassifier()

#TODO: Add ParamGrid 
grid = ParamGridBuilder().addGrid(rf.maxDepth, [3, 5, 8]).build()
evaluator = BinaryClassificationEvaluator()
cv = CrossValidator(estimator=rf, estimatorParamMaps=grid, evaluator=evaluator, numFolds=5)
cvModel = cv.fit(train)
score = evaluator.evaluate(cvModel.transform(train))

#Saving the model
bestModel = cvModel.bestModel
os.system('mkdir rf')
os.chdir(os.getcwd() + '/rf')
bestModel.save(os.getcwd() + '/rfModel')
np.save('score', score)


# TODO: Make predictions on test_transformed and use APK 
'''
prediction = cvModel.transform(test) 
selected = prediction.select("rawPrediction").collect() #TODO: Try "probability" 
predicted_raw =list(map(lambda row: row.rawPrediction, selected))

'''
