'''
Performs full-batch gradient descent with any first-order gradient
optimization method you like on Spark.  You need to specify
gradient calculation by your own, of course; this module cannot
automagically perform symbolic gradient computation.

Remember to add this file into SparkContext.
'''

def _add_losses(r1, r2):
    return r1[0] + r2[0], {c: r1[1][c] + r2[1][c] for c in r1[1]}

def step(dataset, model, grad, desc, count=None, do_desc=True):
    '''
    Perform one full-batch gradient descent step.

    Parameters
    ----------
    dataset : RDD instance of any type
        An RDD of any type.
    model : dict
        A dict of any key-value type.  Usually you want to use parameter names
        as keys and parameter values as dict values.
    grad : callable
        User-provided function which computes per-data-point loss and
        gradient.
        
        It should take two arguments:
        data_point : any
            A data point, which is of the same type as the elements in
            @dataset.
        model : dict
            Your value of @model would be passed into this argument.

        It should return two values:
        loss : number
            A number indicating the loss on this data point.
        grads : dict
            A dict with the same keys as @model, and the values the gradient
            w.r.t. each parameter with the same key.
    desc : callable
        User-provided function which performs the actual descent operation.

        It should take two values:
        model
            Your value of @model would be passed into this argument.  Note
            that the content there would be changed when desc() returns.
        grads
            Your result of @grad() function would be passed into this
            argument.

        In this function you should update your model accordingly by
        updating the values in @model.
    count : int or None
        The number of elements in the dataset.  If given None, step()
        would count the number of elements each time you call it.
    do_desc : bool, default True
        If False, do not perform the desc() operation, only return
        the gradient and loss.

    Returns
    -------
    loss : number
        The average loss over the entire dataset.
    grads : dict
        The average gradient of each parameter over entire dataset.
    '''
    sc = dataset.context
    _model_broadcast = {c: sc.broadcast(model[c]) for c in model}

    if count is None:
        count = dataset.count()
    preds = dataset.map(
            lambda r: grad(r, {c: _model_broadcast[c].value for c in model})
            )
    agg = preds.treeReduce(_add_losses)
    loss = agg[0] / count
    grads = {c: agg[1][c] / count for c in agg[1]}
    if do_desc:
        desc(model, grads)

    for c in _model_broadcast:
        _model_broadcast[c].unpersist(blocking=True)

    return loss, grads
