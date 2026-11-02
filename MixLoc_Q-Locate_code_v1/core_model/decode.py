"""Two-state exact max-score decoder; raw and decoded paths remain separate."""
import numpy as np


def mask_spans(mask):
    padded=np.r_[False,np.asarray(mask,dtype=bool),False].astype(np.int8)
    changes=np.diff(padded)
    return list(zip(np.flatnonzero(changes==1).tolist(),np.flatnonzero(changes==-1).tolist()))


def decode(alpha,a,beta=.25):
    alpha=np.asarray(alpha,dtype=np.float64);a=np.asarray(a,dtype=np.float64)
    n=len(alpha);back=np.zeros((n,2),dtype=np.int8)
    h,m=0.,-np.inf
    for i in range(n):
        # Exact ties select the machine predecessor.
        from_m_h=m+beta*a[i,1]>=h
        from_m_m=m>=h+beta*a[i,0]
        nh=-np.logaddexp(0.,alpha[i])+(m+beta*a[i,1] if from_m_h else h)
        nm=-np.logaddexp(0.,-alpha[i])+(m if from_m_m else h+beta*a[i,0])
        back[i]=(int(from_m_h),int(from_m_m));h,m=nh,nm
    state=int(m+beta*a[n,1]>=h)
    mask=np.zeros(n,dtype=bool)
    for i in range(n-1,-1,-1):
        mask[i]=state;state=int(back[i,state])
    return mask,mask_spans(mask)
