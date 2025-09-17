from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

def dataPreparations():
    sqlite3.connect()



if __name__ == '__main__':
    X, _ = make_blobs(n_samples=300, centers=4, random_state=42, n_features=4)
    print(type(X))



