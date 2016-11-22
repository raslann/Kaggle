
from pyspark import SparkContext
from pyspark.sql import HiveContext, Row
from pyspark.ml.recommendation import *
from util import *

init_spark(verbose_logging=True)
sc.addPyFile('util.py')

df = read_hdfs_csv(sqlContext, 'documents_entities.csv')

def id_map(df, col):
    series = df.select(col).distinct().toPandas()[col]
    return series, dict(zip(series, range(len(series))))

docs, doc_map = id_map(df, 'document_id')
entities, entity_map = id_map(df, 'entity_id')

def mapper(row):
    return Row(document_id=doc_map[row['document_id']],
            entity_id=entity_map[row['entity_id']],
            confidence_level=row['confidence_level'])

df_mapped = df.map(mapper).toDF()

als = ALS(
        rank=100,
        userCol='document_id',
        itemCol='entity_id',
        ratingCol='confidence_level',
        nonnegative=True,
        )
model = als.fit(df_mapped)

model.userFactors.write.parquet('userfactor')
model.itemFactors.write.parquet('itemfactor')
