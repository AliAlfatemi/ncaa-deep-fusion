"""Validated score/rank combination for aligned model predictions."""
import numpy as np
import pandas as pd


def fuse_scores(scores,weights=None,*,method='score',normalize=False):
    values=np.asarray(scores,dtype=float)
    if values.ndim!=2 or min(values.shape)<1 or not np.isfinite(values).all():
        raise ValueError('scores must be a nonempty finite rows-by-models matrix')
    weights=np.ones(values.shape[1]) if weights is None else np.asarray(weights,dtype=float)
    if weights.shape!=(values.shape[1],) or not np.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:
        raise ValueError('weights must match models, be finite/nonnegative, and have positive total')
    if method not in ['score','rank']:raise ValueError('method must be score or rank')
    if method=='rank':
        if len(values)==1:values=np.ones_like(values)
        else:values=1-(pd.DataFrame(values).rank(ascending=False,method='average').to_numpy()-1)/(len(values)-1)
    elif normalize:
        lo=values.min(axis=0);span=np.ptp(values,axis=0)
        values=np.divide(values-lo,span,out=np.full_like(values,0.5),where=span!=0)
    return values@(weights/weights.sum())


def combine_frame(frame,columns,*,id_column='ID',weights=None,method='score',normalize=False):
    if id_column not in frame or frame[id_column].isna().any() or frame[id_column].duplicated().any():
        raise ValueError('each sample needs a unique nonmissing ID')
    if not columns or len(set(columns))!=len(columns) or id_column in columns:
        raise ValueError('choose distinct score columns, excluding the ID')
    absent=set(columns)-set(frame)
    if absent:raise ValueError('missing score columns: '+', '.join(sorted(absent)))
    scores=fuse_scores(frame[columns].to_numpy(),weights,method=method,normalize=normalize)
    out=pd.DataFrame({id_column:frame[id_column].to_numpy(),'score':scores})
    out['rank']=out['score'].rank(ascending=False,method='min').astype(int)
    return out
