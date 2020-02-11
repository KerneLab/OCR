import numpy as np
from kocr import basis


def find(number, points, norm=np.linalg.norm):
    vects = dict([(p, np.array(p)) for p in points])
    dists = dict()
    for a in points:
        for b in points:
            d = norm(vects[a] - vects[b])
            dists[(a, b)] = d
            dists[(b, a)] = d
    apart = []  # 分离点
    cache = dict()  # key点到其余分离点的距离之和
    for point in points:
        if len(apart) < number:
            apart.append(point)
            if len(apart) == number:
                for p in apart:
                    cache[p] = sum([dists[(p, q)] for q in apart if q != p])
        else:
            if point not in cache:
                # 找出离当前节点a最近的现有b
                temp = [(p, cache[p], sum([dists[(point, q)] for q in apart if q != p])) for p in apart]
                temp.sort(key=lambda p: p[2] - p[1], reverse=True)
                for p, old, new in temp:
                    # 如果现有apart集合中存在一个点p，使得给定point到apart点(除p以外)的距离之和更大，则用point替换p点
                    if new > old:
                        idx = apart.index(p)
                        del cache[p]
                        cache[point] = new
                        apart[idx] = point
                        break
    return apart
