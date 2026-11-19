"""Combine aligned model scores without fitting weights on evaluation labels."""
import argparse,json,tempfile
from pathlib import Path
import pandas as pd
from .fusion import combine_frame


def smoke():
    f=pd.DataFrame({'ID':['a','b','c'],'model_a':[0.1,0.5,0.9],'model_b':[0.2,0.4,0.8]})
    out=combine_frame(f,['model_a','model_b'])
    assert out['rank'].tolist()==[3,2,1]
    return {'status':'passed','scope':'synthetic aligned-score fusion check'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True);sub.add_parser('smoke')
    f=sub.add_parser('fuse');f.add_argument('--csv',required=True);f.add_argument('--columns',nargs='+',required=True)
    f.add_argument('--id-column',default='ID');f.add_argument('--output',required=True)
    f.add_argument('--method',choices=['score','rank'],default='score');f.add_argument('--normalize',action='store_true')
    f.add_argument('--weights',nargs='+',type=float)
    a=p.parse_args(argv)
    try:
        if a.command=='smoke':r=smoke()
        else:
            out=Path(a.output)
            if out.exists():raise ValueError('choose a fresh output directory')
            result=combine_frame(pd.read_csv(a.csv),a.columns,id_column=a.id_column,weights=a.weights,method=a.method,normalize=a.normalize)
            out.mkdir(parents=True);result.to_csv(out/'fused_scores.csv',index=False)
            r={'samples':len(result),'method':a.method,'score_columns':a.columns,'weights':a.weights or [1]*len(a.columns),'normalization':a.normalize}
            (out/'metadata.json').write_text(json.dumps(r,indent=2)+'\n')
    except (ValueError,FileNotFoundError) as e:p.error(str(e))
    print(json.dumps(r,indent=2,allow_nan=False))

if __name__=='__main__':main()
