import numpy as np
import Basis


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
    changed = True
    while changed:
        changed = False
        for point in points:
            if len(apart) < number:
                apart.append(point)
                changed = True
                if len(apart) == number:
                    for p in apart:
                        cache[p] = sum([dists[(p, q)] for q in apart if q != p])
            else:
                if point not in cache:
                    # 找出离当前节点a最近的现有b
                    closest_idx = Basis.minindex([dists[(point, p)] for p in apart])
                    closest = apart[closest_idx]
                    tempsum = sum([dists[(point, q)] for q in apart if q != closest])
                    # 如果a与其余点的距离和大于b到其余点的距离，则用a替换b
                    if tempsum > cache[closest]:
                        changed = True
                        del cache[closest]
                        cache[point] = tempsum
                        apart[closest_idx] = point
    return apart
