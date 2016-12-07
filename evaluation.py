import numpy as np

def score(src, dst):
    res = [{}, {}]
    results = []
    
    num_labels = 4

    for i in range(num_labels):
        res[0][i] = set(np.where(src == i)[0])
        res[1][i] = set(np.where(dst == i)[0])
    
    for i in range(num_labels):
        if i not in res[0] or i not in res[1]:
            results.append(1)
            continue
            
        s1 = res[0][i]
        s2 = res[1][i]
        
        inter = len(s1.intersection(s2))
        union = len(s1.union(s2))
        if union == 0:
            results.append(1)
        else:
            results.append(inter / float(union))
        
    return np.mean(results)