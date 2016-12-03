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
train = sqlContext.read.parquet('train_transformed_withpv_normcount')
train = train.withColumn("label", train["label"].cast(DoubleType()))

#Pyspark Random forest algorithm doesn't penalize and adjust for class weights
#We get around by implementing a naive method which is downsampling/filtering out the majority class.
#Further, we incure cost of bias and overfitting due to aritifically balancing our training set. 
#88% of ads appearing in both sets. So, a little bit overfit cost will not be crutial.

train = train.sampleBy('label', fractions={0: .24, 1: 1.0}).cache()  
rf = RandomForestClassifier()


stratified_CV_data = CV_data.sampleBy('Churn', fractions={0: 388./2278, 1: 1.0}).cache()

#TODO: Add ParamGrid 
grid = ParamGridBuilder().addGrid(rf.maxDepth, [3, 5, 8]).build()
evaluator = BinaryClassificationEvaluator()
cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator, numFolds=5)
cvModel = cv.fit(train)
score = evaluator.evaluate(cvModel.transform(train))

#Saving the model
bestModel = cvModel.bestModel
os.system('mkdir rf')
os.chdir(os.getcwd() + '/rf')
bestModel.save(os.getcwd() + '/rfModel')
np.save('score', score)


# TODO: Make predictions on test_transformed and use APK 

