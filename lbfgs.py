
from util import *
from apk import apk
from pyspark.sql import *
from pyspark.mllib.classification import *
from pyspark.mllib.regression import *
from pyspark.mllib.evaluation import *

sc, sqlContext = init_spark(verbose_logging='INFO', show_progress=False)
sc.addPyFile('util.py')
sc.addPyFile('apk.py')

regs = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]

train = sqlContext.read.parquet('train_transformed_withpv')
valid = sqlContext.read.parquet('valid_transformed_withpv')
train_set = train.map(lambda r: LabeledPoint(r['label'], r['features']))

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
    print '#### result check'
    print result.take(5)
    result_group = (
            result
            # map to display_id-row pairs
            .map(lambda r: (r.display_id, [r]))
            # group by display_id
            .reduceByKey(lambda r1, r2: r1 + r2)
            # strip off display_id to get row groups only
            .map(lambda r: r[1])
            )
    print '#### result group check'
    print result_group.take(5)
    result_preds = (
            result_group
            .map(
                lambda g: (
                    # map each row group into ad_id labels and...
                    [r for r in g if r.label == 1],
                    # sorted predictions
                    sorted(g, key=lambda r: r.score, reverse=True)
                    )
                )
            .map(
                lambda r: (
                    [x.ad_id for x in r[0]],
                    [y.ad_id for y in r[1]]
                    )
                )
            )
    print '#### result AP@12 check'
    print result_preds.take(5)
    print '#### MAP@12 = %f' % (
            result_preds.map(lambda r: apk(r[0], r[1], k=12)).mean()
            )
