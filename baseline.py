import pandas as pd
import numpy as np

import sys
rega = float(sys.argv[1])
regb = float(sys.argv[2])    # used to be 10
eval = True

train = pd.read_csv("clicks_actual_train.csv")

if eval:
	valid = pd.read_csv("clicks_validation.csv")
	
	print (valid.shape, train.shape)

cnt = train[train.clicked==1].ad_id.value_counts()
cntall = train.ad_id.value_counts()
del train

def get_prob(k):
    #if k not in cnt:
    #    return 0
    # Laplace Smoothing
    return (cnt.get(k, 0)+rega)/(float(cntall.get(k,0)) + regb)

def srt(x):
    ad_ids = map(int, x.split())
    ad_ids = sorted(ad_ids, key=get_prob, reverse=True)
    return " ".join(map(str,ad_ids))
   
if eval:
	from apk import mapk
	
	y = valid[valid.clicked==1].ad_id.values
	y = [[_] for _ in y]
	p = valid.groupby('display_id').ad_id.apply(list)
	p = [sorted(x, key=get_prob, reverse=True) for x in p]
	
	print (mapk(y, p, k=12))
else:
	subm = pd.read_csv("../input/sample_submission.csv") 
	subm['ad_id'] = subm.ad_id.apply(lambda x: srt(x))
	subm.to_csv("subm_reg_1.csv", index=False)
