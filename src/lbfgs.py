
from util import *
from apk import apk
from pyspark.sql import *
from pyspark.mllib.classification import *
from pyspark.mllib.regression import *
import numpy as NP
import sparkapk

sc, sqlContext = init_spark(verbose_logging='INFO', show_progress=False)
sc.addPyFile('util.py')
sc.addPyFile('apk.py')

regs = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]

train = sqlContext.read.parquet('train_transformed_withpv_withprofile')
valid = sqlContext.read.parquet('valid_transformed_withpv_withprofile')
train_set = train.map(lambda r: LabeledPoint(r['label'], r['features']))
max_mapk = 0
best_reg = 0

for regParam in regs:
    print '==============================='
    print 'Trying reg param %f' % regParam
    print '==============================='
    model = LogisticRegressionWithLBFGS.train(
            train_set,
            regParam=regParam,
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
    map12 = sparkapk.sparkmapk(result, k=12)
    print '#### MAP@12 = %f' % map12
    if max_mapk < map12:
        max_mapk = map12
        best_reg = regParam

print '#### Re-training with best param %f' % best_reg
model = LogisticRegressionWithLBFGS.train(
        train_set, regParam=best_reg, intercept=True
        )
w = model.weights
b = model.intercept
NP.savez('lbfgs-withprofile.npz', w=NP.array(w), b=float(b))
