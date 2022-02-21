# distutils: language=c++
import numpy as np
cimport numpy as np
np.import_array()
from libcpp.vector cimport vector
from libcpp.set cimport set as set_c
from libcpp.pair cimport pair as pair
from libcpp.queue cimport priority_queue
from libcpp cimport bool
ctypedef np.int_t ITYPE_t
ctypedef np.float64_t DTYPE_t
ctypedef bool BTYPE_T
cdef class NSWGraph:
    cdef ITYPE_t dimension
    cdef ITYPE_t regularity
    cdef ITYPE_t guard_hops
    cdef ITYPE_t number_nodes

    cdef DTYPE_t norm_factor
    cdef BTYPE_T quantize_flag
    cdef vector[vector[DTYPE_t]] nodes
    cdef vector[set_c[ITYPE_t]] neighbors

    cdef vector[vector[DTYPE_t]] lookup_table
    cdef vector[DTYPE_t] quantization_values

    cdef DTYPE_t eucl_dist(self, vector[DTYPE_t] v1, vector[DTYPE_t] v2) nogil

    cdef priority_queue[pair[DTYPE_t, ITYPE_t]] delete_duplicate(self, priority_queue[pair[DTYPE_t, ITYPE_t]] queue) nogil

    cdef void search_nsw_basic(self, vector[DTYPE_t] query,
                               set_c[ITYPE_t]* visitedSet,
                               priority_queue[pair[DTYPE_t, ITYPE_t]]* candidates,
                               priority_queue[pair[DTYPE_t, ITYPE_t]]* result,
                               ITYPE_t* res_hops, ITYPE_t top=*, ITYPE_t guard_hops=*) nogil

    cdef ITYPE_t _build_navigable_graph(self, vector[vector[DTYPE_t]] values, ITYPE_t attempts=*) nogil

    cdef pair[vector[ITYPE_t], ITYPE_t] _multi_search(self, vector[DTYPE_t] query, ITYPE_t attempts, ITYPE_t top=*, ITYPE_t guard_hops=*) nogil

    cdef vector[vector[DTYPE_t]] ndarray_to_vector_2(self, np.ndarray array)

    cdef np.ndarray find_quantized_values(self, np.ndarray vector)

    cdef np.ndarray quantize(self, np.ndarray data, ITYPE_t quantization_levels)