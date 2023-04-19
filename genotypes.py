from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])


PC_DARTS_cifar = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))
PC_DARTS_image = Genotype(normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))


PCDARTS = PC_DARTS_cifar

PCDARTS_CAMELYON = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 3), ('dil_conv_3x3', 3), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))

PCDARTS_CAMELYON_FULL = Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 1), ('dil_conv_3x3', 3), ('dil_conv_5x5', 2), ('max_pool_3x3', 4), ('dil_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('sep_conv_5x5', 2), ('sep_conv_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

MTNAS_CAMELYON_FULL = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 3), ('skip_connect', 2), ('avg_pool_3x3', 3), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('avg_pool_3x3', 3), ('sep_conv_3x3', 0), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))

MTNAS_CAMELYON_FULL_B = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 2), ('dil_conv_5x5', 3), ('dil_conv_5x5', 4), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 3), ('max_pool_3x3', 0), ('max_pool_3x3', 4), ('skip_connect', 2)], reduce_concat=range(2, 6))

MTNAS_CAMELYON_FULL_C = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 2), ('sep_conv_5x5', 0), ('max_pool_3x3', 3), ('sep_conv_5x5', 2), ('sep_conv_3x3', 3), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_5x5', 3), ('sep_conv_3x3', 2), ('skip_connect', 3), ('dil_conv_3x3', 4)], reduce_concat=range(2, 6))

MTNAS_CAMELYON_FULL_D = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('skip_connect', 3), ('sep_conv_5x5', 2), ('sep_conv_3x3', 3), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_5x5', 3), ('sep_conv_3x3', 2), ('skip_connect', 3), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

##下面是 reduce_gc

# epoch 32
CAMELYON_C = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 2), ('avg_pool_3x3', 4)], reduce_concat=range(2, 6))
# epoch 37
CAMELYON_B = Genotype(normal=[('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('max_pool_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('skip_connect', 0)], reduce_concat=range(2, 6))
# epoch 34
CAMELYON_A = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('dil_conv_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))

##下面是gc
# epoch 42
CAMELYON_GC_C = Genotype(normal=[('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 1), ('sep_conv_5x5', 3), ('avg_pool_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3), ('max_pool_3x3', 2), ('sep_conv_5x5', 3), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))

# epoch 41
CAMELYON_GC_B = Genotype(normal=[('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 3), ('avg_pool_3x3', 2), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_5x5', 0), ('sep_conv_5x5', 3), ('dil_conv_3x3', 1), ('sep_conv_5x5', 3), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6))

# epoch 37
CAMELYON_GC_A = Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_5x5', 3), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2), ('dil_conv_3x3', 1), ('sep_conv_5x5', 2), ('avg_pool_3x3', 3), ('max_pool_3x3', 4), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))

# 下面是only_reduce
only_reduce = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('skip_connect', 3), ('sep_conv_5x5', 2), ('sep_conv_3x3', 3), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_5x5', 3), ('sep_conv_3x3', 2), ('skip_connect', 3), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))
## 下面是reduce_8 gc 2
# epoch 44
REDUCE_8_A = Genotype(normal=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('skip_connect', 3), ('max_pool_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('max_pool_3x3', 4), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6))

#epoch 49
REDUCE_8_B = Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('skip_connect', 3), ('dil_conv_5x5', 1), ('avg_pool_3x3', 4), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 2), ('dil_conv_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))

## 下面是 reduce 16 gc 1
# epoch 46
REDUCE_16_A = Genotype(normal=[('max_pool_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 2), ('dil_conv_5x5', 3), ('sep_conv_5x5', 2), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 4), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6))

# epoch 43
REDUCE_16_C = Genotype(normal=[('dil_conv_5x5', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 3), ('avg_pool_3x3', 2), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 4), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6))

# epoch 49
REDUCE_16_B = Genotype(normal=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 3), ('avg_pool_3x3', 2), ('skip_connect', 0), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 3), ('avg_pool_3x3', 4), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))
