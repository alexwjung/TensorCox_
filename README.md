# TensorCox

Tensorflow implementation of Coxs' partial likelihood [^1] based on a counting process representation[^2].
Combining Cox's model with the counting process representation allows for more complex censoring patterns as well as the inclusion of time-varying covariates.
Implementing this in Tensorflow with batch optimization methods gives us a powerful approach that can scale to big data problems and make full use of distributed computing environments.
Since the algorithm is fully written in Tensorflow it is easy to integrate it into larger workflows and combine it with Neural Networks etc.

## Installation

 > pip install TensorCox

## Documentation

Documentation for TensorCox is available at https://tensorcox.readthedocs.io/en/latest/

## References

[^1]: Cox, D. R. (1972). Regression models and life‐tables. _Journal of the Royal Statistical Society: Series B (Methodological)_, _34_(2), 187-202.
[^2]: Andersen, P. K., & Gill, R. D. (1982). Cox's regression model for counting processes: a large sample study. _The annals of statistics_, 1100-1120.
