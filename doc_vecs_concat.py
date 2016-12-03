
from util import *
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.mllib.linalg import *

sc, sqlContext = init_spark(verbose_logging='INFO', show_progress=False)
sc.addPyFile('util.py')

doc_vecs = sqlContext.read.parquet('doc_vecs')

doc_vec_columns = [
        'category_vec',
        'entity_vec',
        'topic_vec',
        'source_vec',
        'publisher_vec'
        ]

def mapper_doc(r):
    doc_vecs = [r[c] for c in doc_vec_columns]
    doc_vec = sparse_vector_concat(doc_vecs)
    return Row(document_id=r['document_id'], document_vec=doc_vec)

doc_vecs_concat = doc_vecs.map(mapper_doc).toDF()
doc_vecs_concat.write.parquet('doc_vecs_concat')
