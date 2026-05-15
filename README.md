# Optimizer Collection

This repository contains a curated collection of interesting and novel ideas for query optimization in relational database systems.
Optimizer prototypes are implemented using the [PostBOUND framework](https://github.com/rbergm/PostBOUND).

## Optimizer Prototypes

| Optimizer | Description | Original Paper | Implementation |
| --------- | ----------- | -------------- | -------------- |
| BAO | A reinforcement learning-based plan selector. | Marcus et al.: _BAO: Making Learned Query Optimization Practical_ (SIGMOD'2021) 📄 [Paper](https://doi.org/10.1145/3448016.3452838) | 📁 [Code](https://github.com/Optimizer-Playground/Optimization-Techniques/blob/main/postbound_extensions/bao) |
| FASTgres | A supervised learning-based model for selecting optimizer hints | Woltmann et al.: _FASTgres: Making Learned Query Optimizer Hinting Effective_ (VLDB'2023) 📄 [Paper](https://doi.org/10.14778/3611479.3611528) | 📁 [Code](https://github.com/Optimizer-Playground/Optimization-Techniques/blob/main/postbound_extensions/fastgres) |
| MSCN | A supervised, deep learning-based cardinality estimator. | Kipf et al.: _Learned Cardinalities: Estimating Correlated Joins with Deep Learning._ (CIDR'2019) 📄 [Paper](https://vldb.org/cidrdb/papers/2019/p101-kipf-cidr19.pdf) | 📁 [Code](https://github.com/Optimizer-Playground/Optimization-Techniques/blob/main/postbound_extensions/mscn) |
| SafeBound | An upper bound cardinality estimator based on most common values statistics. | Deeds et al.: _SafeBound: A Practical System for Generating Cardinality Bounds_ (SIGMOD'2023) 📄 [Paper](https://doi.org/10.1145/3588907) | 📁 [Code](https://github.com/Optimizer-Playground/Optimization-Techniques/blob/main/postbound_extensions/safebound) |
| TONIC | A learned (but machine learning-free) operator selection algorithm for joins. | Hertzschuch et al.: _Turbo-Charging SPJ Query Plans with Learned Physical Join Operator Selections._ (VLDB'2022) 📄 [Paper](https://doi.org/10.14778/3551793.3551825) | 📁 [Code](https://github.com/Optimizer-Playground/Optimization-Techniques/blob/main/postbound_extensions/tonic) |
| UES | An upper bound-driven, heuristic join order optimizer. | Hertzschuch et al.: _Simplicity Done Right for Join Ordering._ (CIDR'2021) 📄 [Paper](https://vldb.org/cidrdb/2021/simplicity-done-right-for-join-ordering.html) | 📁 [Code](https://github.com/Optimizer-Playground/Optimization-Techniques/blob/main/postbound_extensions/ues) |

## Progress

- [x] UES
  - [x] baseline implementation
  - [x] documentation
- [x] TONIC
  - [x] baseline implementation
  - [x] documentation
- [x] MSCN
  - [x] baseline implementation
  - [x] documentation
- [x] BAO
  - [x] baseline implementation
  - [x] documentation
- [x] SafeBound
  - [x] baseline implementation
  - [x] documentation
- [ ] FASTgres
  - [x] baseline implementation
  - [ ] documentation
- [ ] general documentation

## Installation

TODO

## Usage

TODO

## Contributing

TODO

This project is currently being maintained by Rico Bergmann ([@rbergm](https://github.com/rbergm/)).

The optimizer implementations have been provided by the following people:

- BAO: provided by Rico Bergmann ([@rbergm](https://github.com/rbergm/))
- FASTgres: provided by Kira Thiessat ([@KiraThiessat](https://github.com/KiraThiessat))
- MSCN: provided by Rico Bergmann ([@rbergm](https://github.com/rbergm/))
- SafeBound: provided by Rico Bergmann ([@rbergm](https://github.com/rbergm/))
- TONIC: provided by Rico Bergmann ([@rbergm](https://github.com/rbergm/))
- UES: provided by Rico Bergmann ([@rbergm](https://github.com/rbergm/))

## Additional Utilities

TODO
