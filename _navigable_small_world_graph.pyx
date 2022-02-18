# distutils: language = c++
#!python
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import random

# import numpy as np
cimport numpy as np
np.import_array()
from libcpp.vector cimport vector
from libcpp.set cimport set as set_c
from libcpp.pair cimport pair as pair
from libc.math cimport sqrt, pow
from libcpp.queue cimport priority_queue
from libc.stdlib cimport rand
import itertools
from libcpp cimport bool
import cython
from cython import declare

DTYPE = np.float64
ITYPE = np.int64

cdef class NSWGraph:
    def __init__(self, ITYPE_t n_nodes, ITYPE_t dimensions, ITYPE_t reg=0, ITYPE_t guard_hops=100):
        self.dimension = dimensions

        self.number_nodes = n_nodes
        self.regularity = self.dimension//2 if reg==0 else reg
        self.guard_hops = guard_hops

        cdef ITYPE_t i

        for i in range(self.number_nodes):
            self.classes.push_back(random.randint(0, 1))

    cdef priority_queue[pair[DTYPE_t, ITYPE_t]] delete_duplicate(self, priority_queue[pair[DTYPE_t, ITYPE_t]] queue):# nogil:
        cdef priority_queue[pair[DTYPE_t, ITYPE_t]] new_que
        cdef set_c[ITYPE_t] tmp_set
        new_que.push(queue.top())
        tmp_set.insert(queue.top().second)
        queue.pop()
        while queue.size() != 0:
            if tmp_set.find(queue.top().second) == tmp_set.end():
                tmp_set.insert(queue.top().second)
                new_que.push(queue.top())
            queue.pop()
        return new_que

    # todo: optimize
    cdef DTYPE_t eucl_dist(self, vector[DTYPE_t] v1, vector[DTYPE_t] v2):
        cdef ITYPE_t i = 0
        cdef DTYPE_t res = 0
        if self.quantize_flag:
            for i in range(v1.size()):
                res += self.lookup_table[int(v2[i])][int(v1[i])]
        else:
            for i in range(v1.size()):
                res += pow(v1[i] - v2[i], 2)
        return res



    cdef void search_nsw_basic(self, vector[DTYPE_t] query,
                               set_c[ITYPE_t]* visitedSet,
                               priority_queue[pair[DTYPE_t, ITYPE_t]]* candidates,
                               priority_queue[pair[DTYPE_t, ITYPE_t]]* result,
                               ITYPE_t* res_hops, ITYPE_t top=5,
                               ITYPE_t guard_hops=100):# nogil:
        cdef ITYPE_t entry = rand() % self.nodes.size()
        cdef ITYPE_t hops = 0
        cdef DTYPE_t closest_dist = 0
        cdef ITYPE_t closest_id = 0
        cdef ITYPE_t e = 0
        cdef DTYPE_t d = 0
        cdef pair[DTYPE_t, ITYPE_t] tmp_pair

        d = self.eucl_dist(query, self.nodes[entry])
        tmp_pair.first = d * (-1)
        tmp_pair.second = entry

        if visitedSet[0].find(entry) == visitedSet[0].end():
            candidates[0].push(tmp_pair)
        tmp_pair.first = tmp_pair.first * (-1)
        result[0].push(tmp_pair)
        hops = 0

        # todo: optimize
        while hops < guard_hops:
            hops += 1
            if candidates[0].size() == 0:
                break
            tmp_pair = candidates[0].top()
            candidates.pop()
            closest_dist = tmp_pair.first * (-1)
            closest_id = tmp_pair.second
            if result[0].size() >= top:
                while result[0].size() > top:
                    result[0].pop()

                if result[0].top().first < closest_dist:
                    break

            #  for every element e from friends of c do:
            for e in self.neighbors[closest_id]:
                # 13 if e is not in visitedSet than
                if visitedSet[0].find(e) == visitedSet[0].end():
                    d = self.eucl_dist(query, self.nodes[e])
                    # 14 add e to visitedSet, candidates, tem pRes
                    visitedSet[0].insert(e)
                    tmp_pair.first = d
                    tmp_pair.second = e
                    result.push(tmp_pair)
                    tmp_pair.first = tmp_pair.first * (-1)
                    candidates.push(tmp_pair)
        res_hops[0] = hops

    def search_nsw_basic_wrapped(self, np.ndarray query, ITYPE_t top=5, ITYPE_t guard_hops=100):
        cdef set_c[ITYPE_t] visitedSet
        cdef priority_queue[pair[DTYPE_t, ITYPE_t]] candidates
        cdef priority_queue[pair[DTYPE_t, ITYPE_t]] result
        cdef vector[ITYPE_t] res
        cdef ITYPE_t hops = 0
        self.search_nsw_basic(query, &visitedSet, &candidates, &result, &hops, top, guard_hops)
        result = self.delete_duplicate(result)
        while result.size() > top:
            result.pop()
        for i in range(result.size()):
            res.push_back(result.top().second)
            result.pop()
        return res, hops

    def knnQueryBatch(self,  ATYPE_t queries,  ITYPE_t attempts=1, ITYPE_t top=5, ITYPE_t guard_hops=100):

        '''knn for batch of queries'''
        result = []
        cdef pair[vector[ITYPE_t], ITYPE_t] res
        for i, query in enumerate(queries):
            if self.quantize_flag:
                # normalized_query = self.norm.transform(query)
                normalized_query = query
                query = self.find_quantized_values(normalized_query)
            res = self._multi_search(query, attempts, top, guard_hops)
            result.append([res.first[::-1]])
        return result

    def knnQuery(self, ATYPE_t query, ITYPE_t attempts=1, ITYPE_t top=5, ITYPE_t guard_hops=100):
        '''knn for single query'''
        if self.quantize_flag:
            # normalized_query = self.norm.transform(query)
            normalized_query = query
            query = self.find_quantized_values(normalized_query)
        cdef pair[vector[ITYPE_t], ITYPE_t] res = self._multi_search(query, attempts, top, guard_hops)
        return res.first[::-1], res.second

    cdef pair[vector[ITYPE_t], ITYPE_t] _multi_search(self, vector[DTYPE_t] query,
                                                      ITYPE_t attempts,
                                                      ITYPE_t top=5, ITYPE_t guard_hops=100):# nogil:

        cdef set_c[ITYPE_t] visitedSet
        cdef priority_queue[pair[DTYPE_t, ITYPE_t]] candidates
        cdef priority_queue[pair[DTYPE_t, ITYPE_t]] result
        cdef vector[ITYPE_t] res
        cdef ITYPE_t i
        cdef ITYPE_t hops
        cdef pair[DTYPE_t, ITYPE_t] j
        cdef ITYPE_t id

        for i in range(attempts):
            self.search_nsw_basic(query, &visitedSet, &candidates, &result, &hops, top=top, guard_hops=guard_hops)
            result = self.delete_duplicate(result)

        while result.size() > top:
            result.pop()
        while res.size() < top:
            el = result.top().second
            res.push_back(el)
            if not result.empty():
                result.pop()
            else:
                break
        return pair[vector[ITYPE_t], ITYPE_t](res, hops)


    cdef ATYPE_t find_quantized_values(self, ATYPE_t vector):
      result = []
      for i, data_value in enumerate(vector):

        result.append((np.abs(self.quantization_values - data_value)).argmin())
      return np.array(result)

    cdef ATYPE_t quantize(self, ATYPE_t data, ITYPE_t quantization_levels):

        self.quantization_values = np.linspace(0, 1, quantization_levels)

        self.lookup_table = np.zeros(shape=(quantization_levels,quantization_levels))
        for v in itertools.combinations(enumerate(self.quantization_values), 2):
            i = v[0][0]
            j = v[1][0]
            self.lookup_table[i][j] = pow(np.abs(v[0][1]-v[1][1]),2)
            self.lookup_table[j][i] = pow(np.abs(v[1][1]-v[0][1]),2)
        quantized_data = []
        for i, vector in enumerate(data):
            quantized_data.append(self.find_quantized_values(vector))
        return np.array(quantized_data)



    def build_navigable_graph(self, ATYPE_t values, ITYPE_t attempts=2, bool quantize=False, ITYPE_t quantization_levels=20):
        self.quantize_flag = quantize
        if self.quantize_flag:
            normalized_values = values
            quantized_data = self.quantize(values, quantization_levels=quantization_levels)
            values = quantized_data


        cdef vector[vector[DTYPE_t]] tmp_result = self.ndarray_to_vector_2(values)
        self._build_navigable_graph(tmp_result, attempts)

    cdef ITYPE_t _build_navigable_graph(self, vector[vector[DTYPE_t]] values, ITYPE_t attempts=1): # nogil:
        cdef vector[DTYPE_t] val
        cdef vector[ITYPE_t] closest
        cdef ITYPE_t c
        cdef ITYPE_t i
        cdef vector[ITYPE_t] res
        cdef set_c[ITYPE_t] tmp_set
        if values.size() != self.number_nodes:
            raise Exception("Number of nodes don't match")
        if values[0].size() != self.dimension:
            raise Exception("Dimension doesn't match")

        self.nodes.push_back(values[0])
        for i in range(self.number_nodes):
            self.neighbors.push_back(tmp_set)

        for i in range(1, self.number_nodes):
            val = values[i]
            closest.clear()
            # search f nearest neighbors of the current value existing in the graph
            closest = self._multi_search(val, attempts, self.regularity, self.guard_hops).first
            # create a new node
            self.nodes.push_back(val)
            # connect the closest nodes to the current node
            for c in closest:
                self.neighbors[i].insert(c)
                self.neighbors[c].insert(i)

    cdef vector[vector[DTYPE_t]] ndarray_to_vector_2(self, np.ndarray array):
        cdef vector[vector[DTYPE_t]] tmp_result
        cdef ITYPE_t i
        for i in range(len(array)):
            tmp_result.push_back((array[i]))
        return tmp_result
