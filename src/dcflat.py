from pyspark import SparkContext
from pyspark.sql import HiveContext
from util import *
from pyspark.sql.types import *

sc = SparkContext()
sc.addPyFile('util.py')
sqlContext = HiveContext(sc)

df = read_hdfs_csv(sqlContext, 'documents_categories.csv')
# Equivalent to SELECT DISTINCT category_id FROM documents_categories
category_id_df = df.select('category_id').distinct()

# Turn the dataframe into a Pandas DataFrame and extract the values into
# a Series
category_ids = category_id_df.toPandas()['category_id']
C = category_ids.count()
# Establish an ID-to-index mapping
category_map = dict(zip(category_ids, range(C)))
# New names for each column holding confidence levels (doesn't really
# matter)
cols = ['category' + str(i) for i in range(C)]

# StructType, StructField, IntegerType and FloatType all come from
# the namespace pyspark.sql.types
# See the official documentation for more details if you are interested.
#
# Usually a schema is defined as a StructType, consists of a list of
# StructField's, each initialized by column name, data type, and whether it
# is nullable, as shown below.
schema = StructType(
        [StructField('document_id', IntegerType(), False)] +
        [StructField(col, FloatType(), False) for col in cols]
        )

def map_row(row):
    global C, category_map
    newrow = [row['document_id']] + [0.] * C
    newrow[category_map[row['category_id']] + 1] = row['confidence_level']
    return newrow
# The toDF() method is undocumented in Spark 1.6.0 documentation.  No
# idea why this happens, but you can view the documentation in interactive
# shell by help().
mapped_df = df.map(map_row).toDF(schema=schema)
reduced_df = mapped_df.groupBy('document_id').sum()

write_hdfs_csv(reduced_df, 'dcflat.csv')
