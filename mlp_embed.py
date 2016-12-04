
from util import *
import sparkgd
import numpy as NP
import numpy.random as RNG
import copy

sc, sqlContext = init_spark(verbose_logging='INFO', show_progress=False)
sc.addPyFile('util.py')
sc.addPyFile('sparkgd.py')
sc.addPyFile('apk.py')

train = sqlContext.read.parquet('train_transformed_withpv_withprofile_noprod_separate')
valid = sqlContext.read.parquet('valid_transformed_withpv_withprofile_noprod_separate')
train_count = train.count()
valid_count = valid.count()

first = train.first()
field_names = [
        "advertiser_vec",
        "campaign_vec",
        "ad_meta_vec",
        "document_vec",
        "ad_document_vec",
        "user_profile"
        ]
assert set(field_names + ['display_id', 'ad_id', 'clicked']) == set(train.columns)
nfields = len(train.columns) - 3    # minus ad_id, display_id, clicked
field_dims = [first[f].size for f in field_names]
print field_dims
embed_dim = 100
hidden_dim = 1000

train = train.rdd.repartition(1000)
valid = valid.rdd.repartition(1000)

U = [RNG.uniform(-0.1, 0.1, (embed_dim, field_dims[i])) for i in range(nfields)]
alpha = [NP.zeros(embed_dim) for _ in range(nfields)]
V = [RNG.uniform(-0.1, 0.1, (hidden_dim, embed_dim)) for _ in range(nfields)]
beta = NP.zeros(hidden_dim)
W = RNG.uniform(-0.1, 0.1, (hidden_dim,))
gamma = NP.array(0, dtype='float')

model = dict(
        [('U' + str(i), U[i]) for i in range(nfields)] +
        [('alpha' + str(i), alpha[i]) for i in range(nfields)] +
        [('V' + str(i), V[i]) for i in range(nfields)] +
        [('beta', beta), ('W', W), ('gamma', gamma)]
        )

def sigmoid(x):
    return 1. / (1 + NP.exp(-x))

def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))

def _trunc_output(output):
    return NP.maximum(NP.minimum(output, 1 - 1e-8), 1e-8)

def log_loss(label, output):
    _output = _trunc_output(output)
    return -(label * NP.log(_output) + (1 - label) * NP.log(1 - _output))

def log_loss_d(label, output):
    if NP.isclose(output, 1) and (label == 1):
        return -1
    elif NP.isclose(output, 0) and (label == 0):
        return 1
    else:
        _output = _trunc_output(output)
        return (output - label) / (output * (1 - output))

def pred(data_point, model):
    Y = data_point.clicked
    X = [data_point[f].toArray() for f in field_names]
    U = [model['U' + str(i)] for i in range(nfields)]
    alpha = [model['alpha' + str(i)] for i in range(nfields)]
    V = [model['V' + str(i)] for i in range(nfields)]
    beta = model['beta']
    W = model['W']
    gamma = model['gamma']

    u0 = [U[i].dot(X[i]) + alpha[i] for i in range(nfields)]
    u = [sigmoid(u0[i]) for i in range(nfields)]
    v0 = sum([V[i].dot(u[i]) for i in range(nfields)]) + beta
    v = sigmoid(v0)
    y0 = W.dot(v) + gamma
    y = sigmoid(y0)

    return y

def grad(data_point, model):
    Y = data_point.clicked
    X = [data_point[f].toArray() for f in field_names]
    U = [model['U' + str(i)] for i in range(nfields)]
    alpha = [model['alpha' + str(i)] for i in range(nfields)]
    V = [model['V' + str(i)] for i in range(nfields)]
    beta = model['beta']
    W = model['W']
    gamma = model['gamma']

    u0 = [U[i].dot(X[i]) + alpha[i] for i in range(nfields)]
    u = [sigmoid(u0[i]) for i in range(nfields)]
    v0 = sum([V[i].dot(u[i]) for i in range(nfields)]) + beta
    v = sigmoid(v0)
    y0 = W.dot(v) + gamma
    y = sigmoid(y0)

    L = log_loss(Y, y)

    dL_dy = log_loss_d(Y, y)
    dL_dy0 = dL_dy * sigmoid_d(y0)
    dL_dW = dL_dy0 * v
    dL_dgamma = dL_dy0
    dL_dv = W.T.dot(dL_dy0)

    dL_dv0 = dL_dv * sigmoid_d(v0)
    dL_dV = [NP.outer(dL_dv0, u[i]) for i in range(nfields)]
    dL_dbeta = dL_dv0
    dL_du = [V[i].T.dot(dL_dv0) for i in range(nfields)]

    dL_du0 = [dL_du[i] * sigmoid_d(u0[i]) for i in range(nfields)]
    dL_dU = [NP.outer(dL_du0[i], X[i]) for i in range(nfields)]
    dL_dalpha = [dL_du0[i] for i in range(nfields)]

    grads = dict(
            [('U' + str(i), dL_dU[i]) for i in range(nfields)] +
            [('alpha' + str(i), dL_dalpha[i]) for i in range(nfields)] +
            [('V' + str(i), dL_dV[i]) for i in range(nfields)] +
            [('beta', dL_dbeta), ('W', dL_dW), ('gamma', dL_dgamma)]
            )

    return L, grads

def desc(model, grads):
    for c in model:
        model[c] -= 0.001 * grads[c]

# The following early-stopping method comes from
# http://deeplearning.net/tutorial/gettingstarted.html
best_valid_loss = NP.inf
best_model = None
patience = 1000
patience_increase = 2
improvement_thres = 0.995

for epoch in range(0, 10000):
    train_loss = sparkgd.step(train, model, grad, desc, train_count)
    valid_loss = sparkgd.step(valid, model, grad, desc, valid_count, do_desc=False)
    print train_loss, valid_loss
    if valid_loss < best_valid_loss:
        if valid_loss < best_valid_loss * improvement_thres:
            patience = max(patience, epoch * patience_increase)
        best_model = copy.deepcopy(model)
        best_valid_loss = valid_loss
    if patience <= epoch:
        break

model_bc = sc.broadcast(best_model)

valid_result = valid.map(
        lambda r: Row(
            ad_id=r.ad_id,
            display_id=r.display_id,
            label=r.label,
            score=pred(r, model_bc.value)
            )
        ).toDF()
valid_result.write.parquet('valid_result_mlp_embed_withpv_withprofile')
