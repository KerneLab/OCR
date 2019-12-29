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
