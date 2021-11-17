# distutils: language = c++
#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import random
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.set cimport set as set_c
from libcpp.pair cimport pair as pair
from libc.math cimport sqrt, pow
from libcpp.queue cimport priority_queue
from libc.stdlib cimport rand


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

    cdef priority_queue[pair[DTYPE_t, ITYPE_t]] delete_duplicate(self, priority_queue[pair[DTYPE_t, ITYPE_t]] queue) nogil:
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

    cdef DTYPE_t eucl_dist(self, vector[DTYPE_t] v1, vector[DTYPE_t] v2) nogil:
        cdef ITYPE_t i = 0
        cdef DTYPE_t res = 0
        for i in range(v1.size()):
            res += pow(v1[i] - v2[i], 2)
        return res

    cdef void search_nsw_basic(self, vector[DTYPE_t] query, set_c[ITYPE_t]* visitedSet, priority_queue[pair[DTYPE_t, ITYPE_t]]* candidates, priority_queue[pair[DTYPE_t, ITYPE_t]]* result, ITYPE_t* res_hops, ITYPE_t top=5, ITYPE_t guard_hops=100) nogil:
        ''' basic algorithm, takes vector query and returns a pair (nearest_neighbours, hops)'''
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
        while hops < guard_hops:
            hops += 1
            if candidates[0].size() == 0:
                break
            # 6 get element c closest from candidates (see paper 4.2.)
            # 7 remove c from candidates
            tmp_pair = candidates[0].top()
            candidates.pop()
            closest_dist = tmp_pair.first * (-1)
            closest_id = tmp_pair.second
            # k-th best of global result
            # new stop condition from paper
            # if c is further than k-th element from result
            # than break repeat
            #! NB this statemrnt from paper will not allow to converge in first run.
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

    def knnQueryBatch(self,  np.ndarray queries,  ITYPE_t attempts=1, ITYPE_t top=5, ITYPE_t guard_hops=100):

        '''knn for batch of queries'''
        result = []
        cdef pair[vector[ITYPE_t], ITYPE_t] res
        for i, query in enumerate(queries):
            res = self._multi_search(query, attempts, top, guard_hops)
            result.append([res.first[::-1]])
        return result

    def knnQuery(self, np.ndarray query, ITYPE_t attempts=1, ITYPE_t top=5, ITYPE_t guard_hops=100):
        '''knn for single query'''
        cdef pair[vector[ITYPE_t], ITYPE_t] res = self._multi_search(query, attempts, top, guard_hops)
        return res.first[::-1], res.second

    cdef pair[vector[ITYPE_t], ITYPE_t] _multi_search(self, vector[DTYPE_t] query, ITYPE_t attempts, ITYPE_t top=5, ITYPE_t guard_hops=100) nogil:
        '''Implementation of `K-NNSearch`, but without keeping the visitedSet'''
        # share visitedSet among searched. Paper, 4.2.p2
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

    def vector_quantization(self, np.ndarray values):
        return values

    def build_navigable_graph(self, np.ndarray values, ITYPE_t attempts=1, bool quantize=False):
        '''Accepts container with values. Returns list with graph nodes'''
        if quantize:
            cdef np.ndarray quantized_values = self.vector_quantization(values)
            cdef vector[vector[DTYPE_t]] tmp_result = self.ndarray_to_vector_2(quantized_values)
        else:
            cdef vector[vector[DTYPE_t]] tmp_result = self.ndarray_to_vector_2(values)

        self._build_navigable_graph(tmp_result, attempts)

    cdef ITYPE_t _build_navigable_graph(self, vector[vector[DTYPE_t]] values, ITYPE_t attempts=1) nogil:
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
        '''Accepts container with values. Returns list with graph nodes'''
        # create graph with one node
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
