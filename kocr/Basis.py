import numpy as np


def center_point(points):
    if len(points) == 0:
        return None
    dims = len(points[0])
    size = len(points)
    return tuple([sum([p[d] for p in points]) / size for d in range(dims)])


def clustering_points(points, maxgap, norm=np.linalg.norm):
    cluster = dict()
    for point in points:
        if len(cluster) == 0:
            cluster[point] = [point]
        else:
            temp = [(i, min([norm(np.array(point) - np.array(p)) for p in group])) for i, group in cluster.items()]
            temp.sort(key=lambda x: x[1])
            i, dist = temp[0]
            if dist <= maxgap:
                cluster[i].append(point)
            else:
                cluster[point] = [point]
    return cluster


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


if __name__ == "__main__":
    ps = [(1, 2), (1, 3)]
    print(center_point(ps))
