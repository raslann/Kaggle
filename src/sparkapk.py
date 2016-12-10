
from apk import apk

def sparkmapk(result, k=12):
    '''
    result : Spark DataFrame with 'ad_id', 'display_id', 'label'
    and 'score' columns.
    '''
    result_group = (
            result
            # map to display_id-row pairs
            .map(lambda r: (r.display_id, [r]))
            # group by display_id
            .reduceByKey(lambda r1, r2: r1 + r2)
            # strip off display_id to get row groups only
            .map(lambda r: r[1])
            )
    result_preds = (
            result_group
            .map(
                lambda g: (
                    # map each row group into ad_id labels and...
                    [r for r in g if r.label == 1],
                    # sorted predictions
                    sorted(g, key=lambda r: r.score, reverse=True)
                    )
                )
            .map(
                lambda r: (
                    [x.ad_id for x in r[0]],
                    [y.ad_id for y in r[1]]
                    )
                )
            )
    map12 = result_preds.map(lambda r: apk(r[0], r[1], k=12)).mean()
    return map12
