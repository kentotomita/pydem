import os

def get_parent(path, n==1):
    if n==1:
        return os.path.split(path)[0]
    else:
        return get_parent(os.path.split(path)[0], n-1)