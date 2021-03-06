from _navigable_small_world_graph import NSWGraph
from sklearn.neighbors import KDTree, BallTree
import faiss
import nmslib
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import psutil
from product_quantization import PQ

def get_process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

def format_bytes(bytes, calc=None):
    if calc=='search':
        return round(abs(bytes)/1e3, 2) # kB
    else: # construction
        return round(abs(bytes) / 1e6, 2)


def calculate_recall(approximated, corrects, query_qty, flag=False):
    recall = 0.0
    for i in range(0, query_qty):
        correct_set = set(corrects[1][i])
        approximated_set = set(approximated[i]) if flag else set(approximated[i][0])
        recall = recall + float(len(correct_set.intersection(approximated_set))) / len(correct_set)
    recall = recall / query_qty
    return round(recall, 2)


def test_index(method: str, data, query, k=100, params=None, space='l2'):
    result = {'method': method}
    true_classifier = NearestNeighbors(n_neighbors=k, metric='l2', algorithm='brute').fit(data)
    corrects = true_classifier.kneighbors(query)

    if method == 'NSWG':
        reg = None
        attempts = 1
        guard_hops = 100
        quantize = False
        levels = 20
        if params:
            reg = params['regularity']
            attempts = params['attempts']
            guard_hops = params['guard_hops']
            quantize = params['quantize']
            levels = params['quantization_levels']

        mem_before = get_process_memory()
        start = time.time()
        index = NSWGraph(n_nodes=len(data), dimensions=len(data[0]), reg=reg, guard_hops=guard_hops)
        index.build_navigable_graph(data, attempts=attempts, quantize=quantize, quantization_levels=levels)
        end = time.time()
        mem_after = get_process_memory()
        result['construction-time'] = round(end - start, 3)
        result['construction-memory'] = format_bytes(mem_after-mem_before, calc='construction')

        mem_before = get_process_memory()
        start = time.time()
        approximated_neigbours = index.knnQueryBatch(query, top=k, guard_hops=guard_hops)
        end = time.time()
        mem_after = get_process_memory()

        result['search-time'] = round(end - start, 3)
        result['search-memory'] = format_bytes(mem_after - mem_before, calc='search')
        result['recall'] = calculate_recall(approximated_neigbours, corrects, query.shape[0])

    elif method == 'kd-tree':
        leaf_size = params['leaf_size'] if params else 2
        mem_before = get_process_memory()
        start = time.time()
        index = KDTree(data, leaf_size=leaf_size)
        end = time.time()
        mem_after = get_process_memory()
        result['construction-time'] = round(end - start, 3)
        result['construction-memory'] = format_bytes(mem_after - mem_before, calc='construction')

        mem_before = get_process_memory()
        start = time.time()
        approximated_neigbours = None
        D, approximated_neigbours = index.query(query, k=k)
        end = time.time()
        mem_after = get_process_memory()
        result['search-time'] = round(end - start, 3)
        result['search-memory'] = format_bytes(mem_after - mem_before, calc='search')

        result['recall'] = calculate_recall(approximated_neigbours, corrects, query.shape[0], True)

    elif method == 'ball-tree':
        leaf_size = params['leaf_size'] if params else 2

        mem_before = get_process_memory()
        start = time.time()
        index = BallTree(data, leaf_size=leaf_size)
        end = time.time()
        mem_after = get_process_memory()
        result['construction-time'] = round(end - start, 3)
        result['construction-memory'] = format_bytes(mem_after - mem_before, calc='construction')

        mem_before = get_process_memory()
        start = time.time()
        approximated_neigbours = None
        D, approximated_neigbours = index.query(query, k=k)
        end = time.time()
        mem_after = get_process_memory()
        result['search-time'] = round(end - start, 3)
        result['search-memory'] = format_bytes(mem_after - mem_before, calc='search')
        result['recall'] = calculate_recall(approximated_neigbours, corrects, query.shape[0], True)
    elif method == 'PQ':
        n_subvectors = params['n_subvectors'] if params else 4
        n_bits_per_vector = params['n_bits'] if params else 8
        save_flag = params['save'] if params else False
        flag_lookup =  params['lookup'] if params else False
        mem_before = get_process_memory()
        start = time.time()
        pq = PQ(n_subvectors=n_subvectors,  n_bits_per_vector=n_bits_per_vector)
        pq.compress(train_data=data, to_save=save_flag)
        end = time.time()
        mem_after = get_process_memory()
        result['construction-time'] = round(end - start, 3)
        result['construction-memory'] = format_bytes(mem_after - mem_before, calc='construction')

        mem_before = get_process_memory()
        start = time.time()
        approximated_neigbours = pq.predict(query, nearest_neighbors=k, using_lookup=flag_lookup)
        end = time.time()
        mem_after = get_process_memory()
        result['search-time'] = round(end - start, 3)
        result['search-memory'] = format_bytes(mem_after - mem_before, calc='search')
        result['recall'] = calculate_recall(approximated_neigbours, corrects, query.shape[0], True)

    else:
        mem_before = get_process_memory()
        start = time.time()
        index = nmslib.init(method=method, space=space, data_type=nmslib.DataType.DENSE_VECTOR)
        index.addDataPointBatch(data)
        index.createIndex(params)
        end = time.time()
        mem_after = get_process_memory()
        result['construction-time'] = round(end - start, 3)
        result['construction-memory'] = format_bytes(mem_after - mem_before, calc='construction')

        mem_before = get_process_memory()
        start = time.time()
        approximated_neigbours = index.knnQueryBatch(query, k=k)
        end = time.time()
        mem_after = get_process_memory()
        result['search-time'] = round(end - start, 3)
        result['search-memory'] = format_bytes(mem_after - mem_before, calc='search')
        result['recall'] = calculate_recall(approximated_neigbours, corrects, query.shape[0])

    return result
