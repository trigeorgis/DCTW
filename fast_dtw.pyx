import numpy.linalg as la
import numpy as np
cimport numpy as np

cdef double min3(double a, double b, double c):
    cdef double m = a
    if b < m:
        m = b
    if c < m:
        m = c
    return m

def backtrack(dtw):
    m, n = np.array(dtw.shape) - 1
    path = []

    while m > 0 and n > 0:
        path.append((m, n))
        
        if dtw[m - 1 , n - 1] <  dtw[m, n - 1] and dtw[m - 1 , n - 1] < dtw[m - 1, n]:
            n -= 1
            m -= 1
        elif dtw[m, n - 1] <  dtw[m - 1, n - 1] and dtw[m, n - 1] < dtw[m - 1, n]:
            n -= 1  
        else:
            m -= 1     

    return np.asarray(path)[::-1] - 1

def dtw_lc(np.ndarray s, np.ndarray t, window_len=4):
    cdef int nrows = s.shape[0]
    cdef int ncols = t.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] dtw = np.zeros((nrows+1, ncols+1), dtype=np.float64)

    window_len = np.max((window_len, abs(nrows - ncols)))

    dtw[...] = np.inf
    dtw[0,0] = 0.0
    
    cdef unsigned int i,j
    cdef np.float64_t cost
    
    for i in range(nrows):
        for j in range(max(0, i - window_len), min(ncols, i + window_len)):
            cost = la.norm(s[i] - t[j])
            dtw[i+1,j+1] = cost + min3(dtw[i,j+1], dtw[i+1,j], dtw[i,j])

    return backtrack(dtw)


def dtw(np.ndarray s, np.ndarray t):
    cdef int nrows = s.shape[0]
    cdef int ncols = t.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] dtw = np.zeros((nrows+1, ncols+1), dtype=np.float64)

    dtw[:,0] = np.inf
    dtw[0,:] = np.inf
    dtw[0,0] = 0.0
    
    cdef unsigned int i,j
    cdef np.float64_t cost
    
    for i in range(nrows):
        for j in range(ncols):
            cost = la.norm(s[i] - t[j])
            dtw[i+1,j+1] = cost + min3(dtw[i,j+1], dtw[i+1,j], dtw[i,j])

    return backtrack(dtw)