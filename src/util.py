# Assumes in pyspark
from pyspark import SparkContext
from pyspark.storagelevel import StorageLevel
from pyspark.sql import HiveContext
from pyspark.sql.types import IntegerType, StructType, StructField
from pyspark.mllib.linalg import Vectors, SparseVector
from collections import OrderedDict
import numpy as NP


def init_spark(verbose_logging=False, show_progress=False):
    if not show_progress:
        SparkContext.setSystemProperty('spark.ui.showConsoleProgress', 'false')
    sc = SparkContext()
    sqlContext = HiveContext(sc)
    if verbose_logging:
        sc.setLogLevel(
                'INFO' if isinstance(verbose_logging, bool)
                else verbose_logging
                )
    return sc, sqlContext


def _build_hdfs_path(path):
    '''
    Prepend the HDFS host and other stuff to given path
    '''
    return 'hdfs://babar.es.its.nyu.edu/user/qg323/' + path


def read_hdfs_csv(sqlContext, filename, header='true'):
    csvreader = (sqlContext
            .read
            .format('com.databricks.spark.csv')
            .options(header=header, inferschema='true')
            )
    return csvreader.load(filename)


def write_hdfs_csv(df, filename, compress=None):
    '''
    Parameters:
        compress: bool
            If True, compress the output to a gzip
    '''
    csvwriter = (
            df.write
            .format('com.databricks.spark.csv')
            .options(header='true')
            )
    if compress:
        csvwriter = csvwriter.options(codec='gzip')
    csvwriter.save(filename)


def nunique(df, col):
    return df.select(col).distinct().count()


def is_none_or_instance(obj, cls):
    return (obj is None) or isinstance(obj, cls)


def persist(rdd):
    '''
    Persists the result of RDD so that we can reuse the result.
    This will turn Spark to compute the result eagerly, rather than
    computing it on-demand.
    '''
    return rdd.persist(StorageLevel.MEMORY_AND_DISK_SER_2)


def sparse_vector_add(v1, v2):
    '''
    Add two SparseVectors.
    Surprisingly PySpark does not provide this functionality.
    '''
    if not (is_none_or_instance(v1, SparseVector) and
            is_none_or_instance(v2, SparseVector)):
        raise TypeError('v1 and v2 are not SparseVectors')
    if v1.size != v2.size:
        raise ValueError('v1 and v2 are not of same size')
    d1 = dict(zip(v1.indices, v1.values))
    d2 = dict(zip(v2.indices, v2.values))
    zero = NP.float64(0)
    indices = sorted(list(set(v1.indices) | set(v2.indices)))
    values = [d1.get(i, zero) + d2.get(i, zero) for i in indices]
    return Vectors.sparse(v1.size, indices, values)


def sparse_vector_mul(v1, v2):
    if not (is_none_or_instance(v1, SparseVector) and
            is_none_or_instance(v2, SparseVector)):
        raise TypeError('v1 and v2 are not SparseVectors')
    if v1.size != v2.size:
        raise ValueError('v1 and v2 are not of same size')
    d1 = dict(zip(v1.indices, v1.values))
    d2 = dict(zip(v2.indices, v2.values))
    indices = sorted(list(set(v1.indices) & set(v2.indices)))
    values = [d1[i] * d2[i] for i in indices]
    return Vectors.sparse(v1.size, indices, values)


def sparse_vector_rshift(v, size, off):
    return Vectors.sparse(size, [i + off for i in v.indices], v.values)


def sparse_vector_concat(vlist):
    size_list = [0] + [v.size for v in vlist]
    off_list = NP.cumsum(size_list)[:-1]
    newsize = sum(size_list)
    newvlist = [sparse_vector_rshift(v, newsize, off)
                for v, off in zip(vlist, off_list)]
    return reduce(sparse_vector_add, newvlist)


def sparse_vector_nmul(v, n):
    return Vectors.sparse(v.size, v.indices, [i * n for i in v.values])


def get_index_map(df, field, index_name):
    field_types = dict(zip(df.schema.names, df.schema.fields))
    # A new instance of the same data type as df.field
    fieldtype = type(field_types[field].dataType)()
    schema = StructType([
            StructField(field, fieldtype, False),
            StructField(index_name, IntegerType(), False)
            ])
    return (df.select(field)
            .distinct()
            .map(lambda r: r[field])
            .zipWithIndex()
            .toDF(schema))


def zip_index(df, key, index):
    indices = get_index_map(df, key, index)
    return df.join(indices, on=key)


def empty_sparse_vector_repr(vec_size):
    return '(%d,[],[])' % vec_size
