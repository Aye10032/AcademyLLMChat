from enum import IntEnum


def get_index_param(_type):
    index_params = {
        IndexType.IVF_FLAT: '{"nlist":16384}',
        IndexType.IVF_SQ8: '{"nlist":16384}',
        IndexType.IVF_PQ: '{"nlist":16384,"m":8,"nbits":8}',
        IndexType.HNSW: '{"M": 8, "efConstruction": 64}',
        IndexType.RHNSW_FLAT: '',
        IndexType.RHNSW_SQ: '',
        IndexType.RHNSW_PQ: '',
        IndexType.IVF_HNSW: '',
        IndexType.ANNOY: '',
        IndexType.AUTOINDEX: ''
    }
    return index_params[_type]


class IndexType(IntEnum):
    IVF_FLAT = 0
    IVF_SQ8 = 1
    IVF_PQ = 2
    HNSW = 3
    RHNSW_FLAT = 4
    RHNSW_SQ = 5
    RHNSW_PQ = 6
    IVF_HNSW = 7
    ANNOY = 8

    AUTOINDEX = 9
