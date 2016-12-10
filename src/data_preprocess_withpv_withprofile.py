
from util import *
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.mllib.linalg import *
from pyspark.mllib.classification import *
from pyspark.mllib.regression import *
from pyspark.ml.feature import *
import numpy as NP

sc, sqlContext = init_spark(verbose_logging='INFO', show_progress=False)
sc.addPyFile('util.py')

clicks_train = read_hdfs_csv(sqlContext, 'clicks_actual_train.csv')
clicks_valid = read_hdfs_csv(sqlContext, 'clicks_validation.csv')
clicks_test = read_hdfs_csv(sqlContext, 'clicks_lab_test.csv')
doc_cat = read_hdfs_csv(sqlContext, 'documents_categories.csv')
doc_ent = read_hdfs_csv(sqlContext, 'documents_entities.csv')
doc_top = read_hdfs_csv(sqlContext, 'documents_topics.csv')
doc_meta = read_hdfs_csv(sqlContext, 'documents_meta.csv')
events = read_hdfs_csv(sqlContext, 'events.csv')
promo = read_hdfs_csv(sqlContext, 'promoted_content.csv')

print '==============================='
print 'Data loaded'
print '==============================='

doc_cat_idx = sqlContext.read.parquet('doc_cat_idx')
doc_ent_idx = sqlContext.read.parquet('doc_ent_idx')
doc_top_idx = sqlContext.read.parquet('doc_top_idx')
doc_idx = sqlContext.read.parquet('doc_idx')
adv_idx = sqlContext.read.parquet('adv_idx')
cam_idx = sqlContext.read.parquet('cam_idx')
ad_idx = sqlContext.read.parquet('ad_idx')
src_idx = sqlContext.read.parquet('src_idx')
pub_idx = sqlContext.read.parquet('pub_idx')
uuid_idx = sqlContext.read.parquet('uuid_idx')

# Sanity checks...
assert doc_cat_idx.count() == doc_cat.count()
assert doc_ent_idx.count() == doc_ent.count()
assert doc_top_idx.count() == doc_top.count()
print '==============================='
print 'Feature index established'
print '==============================='

adv_vecs = sqlContext.read.parquet('adv_vecs')
cam_vecs = sqlContext.read.parquet('cam_vecs')
doc_vecs_concat = sqlContext.read.parquet('doc_vecs_concat')
ad_meta_vecs = sqlContext.read.parquet('ad_meta_vecs')
user_profile = (sqlContext.read.parquet('user_profile')
                .toDF('uuid', 'user_profile'))

print '==============================='
print 'Vectors loaded'
print '==============================='

feature_columns = [
        'advertiser_vec',
        'campaign_vec',
        'ad_meta_vec',
        'document_vec',
        'ad_document_vec',
        'user_profile',
        'user_ad_prod',
        'doc_ad_prod',
        'user_doc_prod',
        ]


def profile_factor_mapper(r):
    row_dict = r.asDict()
    user_doc_prod = sparse_vector_mul(r['user_profile'], r['document_vec'])
    doc_ad_prod = sparse_vector_mul(r['document_vec'], r['ad_document_vec'])
    user_ad_prod = sparse_vector_mul(r['user_profile'], r['ad_document_vec'])
    row_dict['user_ad_prod'] = user_ad_prod
    row_dict['doc_ad_prod'] = doc_ad_prod
    row_dict['user_doc_prod'] = user_doc_prod
    return Row(**row_dict)


def transform_dataset(df):
    newdf = (df
             .join(events, on='display_id')
             .select('uuid', 'document_id', 'ad_id', 'clicked', 'display_id')
             .withColumnRenamed('document_id', 'display_document_id')
             .join(promo, on='ad_id')
             .join(ad_meta_vecs, on='ad_id')
             .join(adv_vecs, on='advertiser_id')
             .drop('advertiser_id')
             .join(doc_vecs_concat, on='document_id')
             .drop('document_id')
             .withColumnRenamed('document_vec', 'ad_document_vec')
             .join(cam_vecs, on='campaign_id')
             .drop('campaign_id')
             .join(user_profile, on='uuid')
             .drop('uuid')
             .withColumnRenamed('display_document_id', 'document_id')
             .join(doc_vecs_concat, on='document_id')
             .drop('document_id'))
    print '###### Schema check'
    print newdf.schema
    newdf = newdf.map(profile_factor_mapper).map(
            lambda r: Row(
                display_id=r['display_id'],
                ad_id=r['ad_id'],
                label=r['clicked'],
                features=sparse_vector_concat([r[c] for c in feature_columns])
                )
            ).toDF()
    return newdf


print '==============================='
print 'Transforming dataset for training'
print '==============================='
train_set = transform_dataset(clicks_train)
train_set.write.parquet('train_transformed_nopv_withprofile')
valid_set = transform_dataset(clicks_valid)
valid_set.write.parquet('valid_transformed_nopv_withprofile')
test_set = transform_dataset(clicks_test)
test_set.write.parquet('test_transformed_nopv_withprofile')
