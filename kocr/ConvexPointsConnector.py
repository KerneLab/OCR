import numpy as np
import Basis


def connect(points, norm=np.linalg.norm):
    path = []
    rest = list(points)
    last = None
    curr = None
    while len(rest) > 0:
        if last is None:
            curr = rest.pop()
        else:
            curr = rest.pop(Basis.minindex([norm(np.array(last) - np.array(p)) for p in rest]))
        last = curr
        path.append(last)
    return path


def path_distances(path, norm=np.linalg.norm):
    last = None
    dists = []
    for p in path:
        if last is not None:
            dists.append(norm(np.array(p) - np.array(last)))
        last = p
    dists.append(norm(np.array(path[0]) - np.array(last)))
    return dists


def rect_shape(corners, norm=np.linalg.norm):
    border_length = path_distances(corners, norm)
    border_length.sort(reverse=True)  # 按边长降序排列
    return int(border_length[2]), int(border_length[0])  # 矩形的短边与长边


def redress_rect_corners(corners, targets, norm=np.linalg.norm):
    """
    矫正矩形的角点
    输入矩形的角点列表与目标矫正点集合，返回源角点对应的矫正坐标列表
    """
    source_corners = list(map(
        lambda target: corners[Basis.minindex([norm(np.array(target) - np.array(p)) for p in corners])],
        targets))  # 寻找离矫正点最近的源点
    mapper = dict(zip(source_corners, targets))
    return [mapper[p] for p in corners]


if __name__ == "__main__":
    pass
