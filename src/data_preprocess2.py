
from util import *
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.mllib.linalg import *
from pyspark.mllib.classification import *
from pyspark.mllib.regression import *
from pyspark.ml.feature import *
import numpy as NP

sc, sqlContext = init_spark(verbose_logging='DEBUG', show_progress=False)
sc.addPyFile('util.py')

clicks_train = read_hdfs_csv(sqlContext, 'clicks_actual_train.csv')
clicks_valid = read_hdfs_csv(sqlContext, 'clicks_validation.csv')
clicks_test = read_hdfs_csv(sqlContext, 'clicks_lab_test.csv')
doc_cat = read_hdfs_csv(sqlContext, 'documents_categories.csv')
doc_ent = read_hdfs_csv(sqlContext, 'documents_entities.csv')
doc_top = read_hdfs_csv(sqlContext, 'documents_topics.csv')
doc_meta = read_hdfs_csv(sqlContext, 'documents_meta.csv')
events = read_hdfs_csv(sqlContext, 'events.csv')
promo = read_hdfs_csv(sqlContext, 'promoted_content.csv').drop('document_id')

print '==============================='
print 'Data loaded'
print '==============================='

# Each sample contains the following features in this order:
# User ID
# Document category
# Document topic
# Document entity
# Ad publisher ID
# Ad campaign ID
# Document source ID
# Document publisher ID

# Compute the offset for each feature on the sparse vector.
num_users = nunique(events, events.uuid)
num_cats = nunique(doc_cat, doc_cat.category_id)
num_ents = nunique(doc_ent, doc_ent.entity_id)
num_tops = nunique(doc_top, doc_top.topic_id)
num_advs = nunique(promo, promo.advertiser_id)
num_cams = nunique(promo, promo.campaign_id)
num_srcs = nunique(doc_meta, doc_meta.source_id)
num_pubs = nunique(doc_meta, doc_meta.publisher_id)

(off_users, off_cats, off_ents, off_tops, off_advs, off_cams,
    off_srcs, off_pubs, vecsize) = (
        NP.cumsum([
            0, num_users, num_cats, num_ents, num_tops, num_advs, num_cams,
            num_srcs, num_pubs
            ])
        )

print '==============================='
print 'Users: %d' % num_users, off_users
print 'Categories: %d' % num_cats, off_cats
print 'Entities: %d' % num_ents, off_ents
print 'Topics: %d' % num_tops, off_tops
print 'Advertisers: %d' % num_advs, off_advs
print 'Campaigns: %d' % num_cams, off_cams
print 'Sources: %d' % num_srcs, off_srcs
print 'Publishers: %d' % num_pubs, off_pubs
print '==============================='


uuid_vecs = sqlContext.read.parquet('uuid_vecs')
adv_vecs = sqlContext.read.parquet('adv_vecs')
cam_vecs = sqlContext.read.parquet('cam_vecs')
doc_vecs = sqlContext.read.parquet('doc_vecs')


def vecsum(r):
    vec = r['uuid_vec']
    vec = sparse_vector_add(vec, r['category_vec'])
    vec = sparse_vector_add(vec, r['entity_vec'])
    vec = sparse_vector_add(vec, r['topic_vec'])
    vec = sparse_vector_add(vec, r['advertiser_vec'])
    vec = sparse_vector_add(vec, r['campaign_vec'])
    vec = sparse_vector_add(vec, r['source_vec'])
    vec = sparse_vector_add(vec, r['publisher_vec'])
    return r['display_id'], r['ad_id'], r['clicked'], vec


def transform_dataset(df):
    newdf = (df
             .join(events, on='display_id')
             .select('display_id', 'uuid', 'document_id', 'ad_id', 'clicked')
             .join(promo, on='ad_id')
             .join(adv_vecs, on='advertiser_id')
             .drop('advertiser_id')
             .join(cam_vecs, on='campaign_id')
             .drop('campaign_id')
             .join(uuid_vecs, on='uuid')
             .drop('uuid')
             .join(doc_vecs, on='document_id')
             .drop('document_id'))
    print '###### Schema check'
    print newdf.schema
    return newdf.map(vecsum)


def transform_dataset_to_df(df):
    return (transform_dataset(df)
            .toDF()
            .withColumnRenamed('_1', 'display_id')
            .withColumnRenamed('_2', 'ad_id')
            .withColumnRenamed('_3', 'label')
            .withColumnRenamed('_4', 'features'))


print '==============================='
print 'Transforming dataset for training'
print '==============================='
valid_set = transform_dataset_to_df(clicks_valid)
valid_set.write.parquet('valid_transformed')
test_set = transform_dataset_to_df(clicks_test)
test_set.write.parquet('test_transformed')
