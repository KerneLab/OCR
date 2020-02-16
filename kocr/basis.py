import numpy as np
import cv2


def center_point(points):
    if len(points) == 0:
        return None
    dims = len(points[0])
    size = len(points)
    return tuple([sum([p[d] for p in points]) / size for d in range(dims)])


def clustering_points(points, max_gap,
                      norm=np.linalg.norm,
                      center_trans=lambda x: int(round(x))):
    cluster = {}
    for point in points:
        if len(cluster) == 0:
            cluster[point] = [point]
        else:
            temp = [(i, min([norm(np.array(point) - np.array(p)) for p in group])) for i, group in cluster.items()]
            temp.sort(key=lambda d: d[1])
            i, dist = temp[0]
            if dist <= max_gap:
                cluster[i].append(point)
            else:
                cluster[point] = [point]
    for g, s in cluster.items():
        c = center_point(s)
        del cluster[g]
        cluster[tuple([center_trans(i) for i in list(c)])] = s
    return cluster


def dict_get(d, key, default):
    if key not in d:
        res = default()
        d[key] = res
        return res
    else:
        return d[key]


def dist_point_line(point, line,
                    is_line_segment=True,
                    norm=np.linalg.norm):
    p = np.array(point)
    a, b = [np.array(end) for end in line]
    ab = b - a
    ap = p - a
    d = norm(ab)
    r = 0 if d == 0 else ab.dot(ap) / (d ** 2)
    ac = r * ab
    if is_line_segment:
        if r <= 0:
            return norm(ap)
        elif r >= 1:
            return norm(p - b)
        else:
            return norm(ap - ac)
    else:
        return norm(ap - ac)


def flatten(coll):
    flat = []
    for e in coll:
        flat.extend(e)
    return flat


def groupby(coll, key):
    res = {}
    for e in coll:
        k = key(e)
        dict_get(res, k, lambda: []).append(e)
    return res


def group_reverse_map(group_res,
                      value=lambda v: v, key=lambda k: k):
    """
    Convert
    {
        k1:[u1,u2,...],
        k2:[v1,v2,...],
        ...
    }
    To
    {
        u1:k1,
        u2:k1,
        ...,
        v1:k2,
        v2:k2,
        ...
    }
    """
    return dict([(value(v), key(g)) for g, l in group_res.items() for v in l])


def imshow(img, name=''):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(name)


def maxindex(coll):
    """
    返回集合中最大值的下标
    """
    return None if len(coll) == 0 else coll.index(max(coll))


def minindex(coll):
    """
    返回集合中最小值的下标
    """
    return None if len(coll) == 0 else coll.index(min(coll))


def polygon_to_box(polygon):
    print(polygon)
    return (polygon[0], polygon[3], polygon[4], polygon[7])


def sort(coll, key=lambda x: x, reverse=False):
    coll.sort(key=key, reverse=reverse)
    return coll


if __name__ == '__main__':
    print(flatten([[1, 2], [], [5]]))
