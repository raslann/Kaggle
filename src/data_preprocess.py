
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


# Add a column for each document table representing the category/entity/topic
# index.
doc_cat_idx = persist(zip_index(doc_cat, 'category_id', 'category_idx'))
doc_ent_idx = persist(zip_index(doc_ent, 'entity_id', 'entity_idx'))
doc_top_idx = persist(zip_index(doc_top, 'topic_id', 'topic_idx'))
uuid_idx = persist(get_index_map(events, 'uuid', 'uuid_idx'))
adv_idx = persist(get_index_map(promo, 'advertiser_id', 'advertiser_idx'))
cam_idx = persist(get_index_map(promo, 'campaign_id', 'campaign_idx'))
src_idx = persist(get_index_map(doc_meta, 'source_id', 'source_idx'))
pub_idx = persist(get_index_map(doc_meta, 'publisher_id', 'publisher_idx'))
# Sanity checks...
assert doc_cat_idx.count() == doc_cat.count()
assert doc_ent_idx.count() == doc_ent.count()
assert doc_top_idx.count() == doc_top.count()
print '==============================='
print 'Feature index established'
print '==============================='


def one_hot_encode_mapper(r, size, off, stringify=False):
    vec = Vectors.sparse(size, [off + r[1]], [1])
    return vec if not stringify else Vectors.stringify(vec)


def one_hot_encode(df, size, off, key, vec_name):
    return (df.map(lambda r: (r[0], Vectors.sparse(size, [off + r[1]], [1])))
            .toDF()
            .withColumnRenamed('_1', key)
            .withColumnRenamed('_2', vec_name))


def one_hot_stringify(df, size, off, key, vec_name):
    return (df
            .map(
                lambda r: (
                    r[0],
                    Vectors.stringify(Vectors.sparse(size, [off + r[1]], [1]))
                    )
                )
            .toDF()
            .withColumnRenamed('_1', key)
            .withColumnRenamed('_2', vec_name))


# Other one-hot encodings
uuid_vecs = one_hot_encode(uuid_idx, vecsize, off_users, 'uuid', 'uuid_vec')
adv_vecs = one_hot_encode(adv_idx, vecsize, off_advs, 'advertiser_id',
                          'advertiser_vec')
cam_vecs = one_hot_encode(cam_idx, vecsize, off_cams, 'campaign_id',
                          'campaign_vec')
src_vecs = one_hot_stringify(src_idx, vecsize, off_srcs, 'source_id',
                             'source_vec')
pub_vecs = one_hot_stringify(pub_idx, vecsize, off_pubs, 'publisher_id',
                             'publisher_vec')


# Encode documents into SparseVector's.
def doc_mapper(row, vec_size, vec_off, key, index, value):
    vec = Vectors.sparse(
            vec_size,
            [vec_off + row[index]],
            [row[value]]
            )
    return (row[key], vec)


def doc_encoder(df, vec_size, vec_off, key, index, value, vec_name):
    mapped = df.map(
            lambda r: doc_mapper(r, vec_size, vec_off, key, index, value)
            )
    newdf = (mapped
             .reduceByKey(sparse_vector_add)
             .map(lambda r: (r[0], Vectors.stringify(r[1])))
             .toDF())
    return (newdf
            .withColumnRenamed('_1', key)
            .withColumnRenamed('_2', vec_name))

doc_cat_vec = doc_encoder(
        doc_cat_idx, vecsize, off_cats, 'document_id',
        'category_idx', 'confidence_level', 'category_vec')
doc_ent_vec = doc_encoder(
        doc_ent_idx, vecsize, off_ents, 'document_id',
        'entity_idx', 'confidence_level', 'entity_vec')
doc_top_vec = doc_encoder(
        doc_top_idx, vecsize, off_tops, 'document_id',
        'topic_idx', 'confidence_level', 'topic_vec')

doc_vecs = (doc_meta
            .drop('publish_time')
            .join(src_vecs, on='source_id')
            .drop('source_id')
            .join(pub_vecs, on='publisher_id')
            .drop('publisher_id')
            .join(doc_cat_vec, on='document_id', how='outer')
            .join(doc_ent_vec, on='document_id', how='outer')
            .join(doc_top_vec, on='document_id', how='outer'))
print '###### Schema check'
print doc_vecs.schema
doc_vecs = (
        doc_vecs
        .fillna(empty_sparse_vector_repr(vecsize))
        .map(
                lambda r: tuple(
                    [r[0]] +
                    [Vectors.parse(r[i]) for i in range(1, 6)]
                    )
        ).toDF()
        .withColumnRenamed('_1', 'document_id')
        .withColumnRenamed('_2', 'source_vec')
        .withColumnRenamed('_3', 'publisher_vec')
        .withColumnRenamed('_4', 'category_vec')
        .withColumnRenamed('_5', 'entity_vec')
        .withColumnRenamed('_6', 'topic_vec')
        )
print '==============================='
print 'Document sparse vector encoded'
print '==============================='
adv_vecs.write.parquet('adv_vecs')
cam_vecs.write.parquet('cam_vecs')
uuid_vecs.write.parquet('uuid_vecs')
doc_vecs.write.parquet('doc_vecs')
print '==============================='
print 'Encoding saved'
print '==============================='


def vecsum(r):
    vec = r['uuid_vec']
    vec = sparse_vector_add(vec, r['category_vec'])
    vec = sparse_vector_add(vec, r['entity_vec'])
    vec = sparse_vector_add(vec, r['topic_vec'])
    vec = sparse_vector_add(vec, r['advertiser_vec'])
    vec = sparse_vector_add(vec, r['campaign_vec'])
    vec = sparse_vector_add(vec, r['source_vec'])
    vec = sparse_vector_add(vec, r['publisher_vec'])
    return r['clicked'], vec


def transform_dataset(df):
    newdf = (df
             .join(events, on='display_id')
             .select('uuid', 'document_id', 'ad_id', 'clicked')
             .join(promo, on='ad_id')
             .drop('ad_id')
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
            .withColumnRenamed('_1', 'label')
            .withColumnRenamed('_2', 'features'))


print '==============================='
print 'Transforming dataset for training'
print '==============================='
train_set = transform_dataset_to_df(clicks_train)
train_set.write.parquet('train_transformed')
valid_set = transform_dataset_to_df(clicks_valid)
valid_set.write.parquet('valid_transformed')
test_set = transform_dataset_to_df(clicks_test)
test_set.write.parquet('test_transformed')
