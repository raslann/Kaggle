
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
# Number of clicks on an Ad
# Number of browses on an Ad
# Click/browse ratio of an Ad

# Compute the offset for each feature on the sparse vector.
num_docs = nunique(doc_meta, doc_meta.document_id)
num_cats = nunique(doc_cat, doc_cat.category_id)
num_ents = nunique(doc_ent, doc_ent.entity_id)
num_tops = nunique(doc_top, doc_top.topic_id)
num_advs = nunique(promo, promo.advertiser_id)
num_cams = nunique(promo, promo.campaign_id)
num_srcs = nunique(doc_meta, doc_meta.source_id)
num_pubs = nunique(doc_meta, doc_meta.publisher_id)
num_ad_meta = 3

(off_docs, off_cats, off_ents, off_tops, off_advs, off_cams,
    off_srcs, off_pubs, off_ad_meta, vecsize) = (
        NP.cumsum([
            0, num_docs, num_cats, num_ents, num_tops, num_advs, num_cams,
            num_srcs, num_pubs, num_ad_meta
            ])
        )

print '==============================='
print 'Documents: %d' % num_docs, off_docs
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

new_idx = False

if new_idx:
    doc_cat_idx = persist(zip_index(doc_cat, 'category_id', 'category_idx'))
    doc_ent_idx = persist(zip_index(doc_ent, 'entity_id', 'entity_idx'))
    doc_top_idx = persist(zip_index(doc_top, 'topic_id', 'topic_idx'))
    doc_idx = persist(get_index_map(doc_meta, 'document_id', 'document_idx'))
    adv_idx = persist(get_index_map(promo, 'advertiser_id', 'advertiser_idx'))
    cam_idx = persist(get_index_map(promo, 'campaign_id', 'campaign_idx'))
    ad_idx = persist(get_index_map(promo, 'ad_id', 'ad_idx'))
    src_idx = persist(get_index_map(doc_meta, 'source_id', 'source_idx'))
    pub_idx = persist(get_index_map(doc_meta, 'publisher_id', 'publisher_idx'))
    uuid_idx = persist(get_index_map(page_views, 'uuid', 'uuid_idx'))

    doc_cat_idx.write.parquet('doc_cat_idx')
    doc_ent_idx.write.parquet('doc_ent_idx')
    doc_top_idx.write.parquet('doc_top_idx')
    doc_idx.write.parquet('doc_idx')
    adv_idx.write.parquet('adv_idx')
    cam_idx.write.parquet('cam_idx')
    ad_idx.write.parquet('ad_idx')
    src_idx.write.parquet('src_idx')
    pub_idx.write.parquet('pub_idx')
    uuid_idx.write.parquet('uuid_idx')
else:
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

adv_vecs = one_hot_encode(adv_idx, num_advs, 0, 'advertiser_id',
                          'advertiser_vec')
cam_vecs = one_hot_encode(cam_idx, num_cams, 0, 'campaign_id',
                          'campaign_vec')
src_vecs = one_hot_stringify(src_idx, num_srcs, 0, 'source_id',
                             'source_vec')
pub_vecs = one_hot_stringify(pub_idx, num_pubs, 0, 'publisher_id',
                             'publisher_vec')

ad_clicks = (clicks_train.groupBy('ad_id').sum('clicked')
             .toDF('ad_id', 'total_clicks'))
ad_browses = (clicks_train.groupBy('ad_id').count()
              .toDF('ad_id', 'total_browses'))
ad_browsed_meta = (
        ad_clicks
        .join(ad_browses, on='ad_id')
        .map(
            lambda r: Row(
                ad_id=r['ad_id'],
                total_clicks=r['total_clicks'],
                total_browses=r['total_browses'],
                click_rate=(
                    (float(r['total_clicks']) / float(r['total_browses']))
                    if r['total_browses'] != 0 else 0
                    )
                )
            )
        .toDF()
        )
ad_meta = (
        promo
        .select('ad_id')
        .join(ad_browsed_meta, on='ad_id', how='left')
        .fillna(0)
        )
ad_meta_vecs = (
        ad_meta
        .map(
            lambda r: Row(
                ad_id=r['ad_id'],
                ad_meta_vec=Vectors.sparse(
                    3,
                    [0, 1, 2],
                    [r['total_clicks'], r['total_browses'], r['click_rate']]
                    )
                )
            )
        .toDF()
        )
assert ad_meta_vecs.count() == ad_idx.count()


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
        doc_cat_idx, num_cats, 0, 'document_id',
        'category_idx', 'confidence_level', 'category_vec')
doc_ent_vec = doc_encoder(
        doc_ent_idx, num_ents, 0, 'document_id',
        'entity_idx', 'confidence_level', 'entity_vec')
doc_top_vec = doc_encoder(
        doc_top_idx, num_tops, 0, 'document_id',
        'topic_idx', 'confidence_level', 'topic_vec')

doc_vecs = (doc_meta
            .drop('publish_time')
            .join(src_vecs, on='source_id', how='left')
            .drop('source_id'))
doc_vecs = (doc_vecs
            .join(pub_vecs, on='publisher_id', how='left')
            .drop('publisher_id'))
doc_vecs = (doc_vecs
            .join(doc_cat_vec, on='document_id', how='left'))
doc_vecs = (doc_vecs
            .join(doc_ent_vec, on='document_id', how='left'))
doc_vecs = (doc_vecs
            .join(doc_top_vec, on='document_id', how='left'))
print '###### Schema check'
print doc_vecs.schema
doc_vecs = (
        doc_vecs
        .fillna({
            'source_vec': empty_sparse_vector_repr(num_srcs),
            'publisher_vec': empty_sparse_vector_repr(num_pubs),
            'category_vec': empty_sparse_vector_repr(num_cats),
            'entity_vec': empty_sparse_vector_repr(num_ents),
            'topic_vec': empty_sparse_vector_repr(num_tops),
            })
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

page_views = read_hdfs_csv(sqlContext, 'page_views.csv')
page_views_idx = page_views.join(doc_idx, on='document_id')
page_views_idx = page_views_idx.join(uuid_idx, on='uuid')

page_views_count = page_views_idx.groupby(
        'uuid', 'uuid_idx', 'document_id', 'document_idx'
        ).count()
page_views_mapped = page_views_count.map(
        lambda r: (
            r['uuid'],
            Vectors.sparse(
                num_docs,
                [r['document_idx']],
                [r['count']]
                )
            )
        )
page_views_reduced = page_views_mapped.reduceByKey(sparse_vector_add)
uuid_vecs = page_views_reduced.toDF().toDF('uuid', 'uuid_vec')

adv_vecs.write.parquet('adv_vecs')
cam_vecs.write.parquet('cam_vecs')
uuid_vecs.write.parquet('uuid_vecs')
assert uuid_vecs.count() == uuid_idx.count()
doc_vecs.write.parquet('doc_vecs')
page_views_count.write.parquet('page_views_count')
ad_meta_vecs.write.parquet('ad_meta_vecs')
print '==============================='
print 'Encoding saved'
print '==============================='
