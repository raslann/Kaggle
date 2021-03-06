
from util import *
from apk import apk
from pyspark.sql import *
from pyspark.mllib.classification import *
from pyspark.mllib.regression import *
from pyspark.mllib.evaluation import *

sc, sqlContext = init_spark(verbose_logging='INFO', show_progress=False)
sc.addPyFile('util.py')
sc.addPyFile('apk.py')

# Models selected:
# Logistic regression with L2 Reg coef 1e-2 (0.6498)
regs = 1e-5

train = sqlContext.read.parquet('train_transformed_withpv')
valid = sqlContext.read.parquet('valid_transformed_withpv')
train_set = train.map(lambda r: LabeledPoint(r['label'], r['features']))

model = LogisticRegressionWithLBFGS.train(
        train_set,
        regParam=regs,
        intercept=True
        )
w = model.weights
b = model.intercept
result = valid.map(
        lambda r: Row(
            ad_id=r['ad_id'],
            display_id=r['display_id'],
            label=r['label'],
            score=float(r['features'].dot(w) + float(b))
            )
        ).toDF()
result.write.parquet('lbfgs_l2_reg1e-5')
