# Optimizer Collection

This repository contains a curated collection of interesting and novel ideas for query optimization in relational database systems.
Optimizer prototypes are implemented using the [PostBOUND framework](https://github.com/rbergm/PostBOUND).

## Optimizer Prototypes

| Optimizer | Description | Original Paper | Implementation |
| --------- | ----------- | -------------- | -------------- |
| UES | An upper bound-driven, heuristic join order optimizer. | Hertzschuch et al.: _Simplicity Done Right for Join Ordering._ (CIDR'2021) [🌐 Link](https://vldb.org/cidrdb/2021/simplicity-done-right-for-join-ordering.html) | [📁 Code](https://github.com/Optimizer-Playground/Optimization-Techniques/blob/main/postbound_extensions/ues) |
| TONIC | A learned (but machine learning-free) operator selection algorithm for joins. | Hertzschuch et al.: _Turbo-Charging SPJ Query Plans with Learned Physical Join Operator Selections._ (VLDB'2022) [🌐 Link](https://doi.org/10.14778/3551793.3551825) | [📁 Code](https://github.com/Optimizer-Playground/Optimization-Techniques/blob/main/postbound_extensions/tonic) |
| MSCN | A supervised, deep learning-based cardinality estimator. | Kipf et al.: _Learned Cardinalities: Estimating Correlated Joins with Deep Learning._ (CIDR'2019) [🌐 Link](https://vldb.org/cidrdb/papers/2019/p101-kipf-cidr19.pdf) | [📁 Code](https://github.com/Optimizer-Playground/Optimization-Techniques/blob/main/postbound_extensions/mscn) |

## Installation

TODO

## Usage

TODO

## Contributing

TODO

## Additional Utilities

TODO
