
from util import *
from pyspark.mllib.recommendation import *
from pyspark.mllib.linalg import *
from pyspark.sql import *

sc, sqlContext = init_spark(verbose_logging='INFO', show_progress=False)

pvc = sqlContext.read.parquet('page_views_count')

# Default 200 partitions overrun memory
upm = (pvc.map(lambda r: (r['uuid_idx'], r['document_idx'], r['count']))
       .repartition(500))

model = ALS.trainImplicit(upm, 100, alpha=40.)

user_factors = model.userFeatures().map(
        lambda r: Row(uuid_idx=r[0], uuid_factor=Vectors.dense(r[1]))
        ).toDF()
doc_factors = model.productFeatures().map(
        lambda r: Row(document_idx=r[0], document_factor=Vectors.dense(r[1]))
        ).toDF()

user_factors.write.parquet('uuid_factors')
doc_factors.write.parquet('doc_factors')
