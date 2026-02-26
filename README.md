# Optimizer Collection

This repository contains a curated collection of interesting and novel ideas for query optimization in relational database systems.
Optimizer prototypes are implemented using the [PostBOUND framework](https://github.com/rbergm/PostBOUND).

## Optimizer Prototypes

| Optimizer | Description | Original Paper | Implementation |
| --------- | ----------- | -------------- | -------------- |
| BAO | A reinforcement learning-based plan selector. | Marcus et al.: _BAO: Making Learned Query Optimization Practical_ (SIGMOD'2021) 📄 [Link](https://doi.org/10.1145/3448016.3452838) | 📁 [Code](https://github.com/Optimizer-Playground/Optimization-Techniques/blob/main/postbound_extensions/bao) |
| MSCN | A supervised, deep learning-based cardinality estimator. | Kipf et al.: _Learned Cardinalities: Estimating Correlated Joins with Deep Learning._ (CIDR'2019) 📄 [Link](https://vldb.org/cidrdb/papers/2019/p101-kipf-cidr19.pdf) | 📁 [Code](https://github.com/Optimizer-Playground/Optimization-Techniques/blob/main/postbound_extensions/mscn) |
| SafeBound | An upper bound cardinality estimator based on most common values statistics. | Deeds et al.: _SafeBound: A Practical System for Generating Cardinality Bounds_ (SIGMOD'2023) 📄 [Link](https://doi.org/10.1145/3588907) | 📁 [Code](https://github.com/Optimizer-Playground/Optimization-Techniques/blob/main/postbound_extensions/safebound) |
| TONIC | A learned (but machine learning-free) operator selection algorithm for joins. | Hertzschuch et al.: _Turbo-Charging SPJ Query Plans with Learned Physical Join Operator Selections._ (VLDB'2022) 📄 [Link](https://doi.org/10.14778/3551793.3551825) | 📁 [Code](https://github.com/Optimizer-Playground/Optimization-Techniques/blob/main/postbound_extensions/tonic) |
| UES | An upper bound-driven, heuristic join order optimizer. | Hertzschuch et al.: _Simplicity Done Right for Join Ordering._ (CIDR'2021) 📄 [Link](https://vldb.org/cidrdb/2021/simplicity-done-right-for-join-ordering.html) | 📁 [Code](https://github.com/Optimizer-Playground/Optimization-Techniques/blob/main/postbound_extensions/ues) |

## Progress

- [ ] UES
  - [x] baseline implementation
  - [ ] documentation
- [ ] TONIC
  - [x] baseline implementation
  - [ ] documentation
- [ ] MSCN
  - [x] baseline implementation
  - [ ] documentation
- [ ] BAO
  - [x] baseline implementation
  - [ ] documentation
- [ ] SafeBound
  - [ ] baseline implementation
  - [ ] documentation
- [ ] documentation

## Installation

TODO

## Usage

TODO

## Contributing

TODO

## Additional Utilities

TODO
