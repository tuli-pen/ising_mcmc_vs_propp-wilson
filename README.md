# Ising Model: Markov Chain Monte Carlo (MCMC) VS Propp–Wilson (Coupling From The Past)
This repository contains a project for the course “Cadenas de Markov & Aplicaciones". This project compares two sampling methods for the Ising model on a K×K lattice: the Gibbs Sampler algorithm and the Coupling From The Past algorithm.

### a) MCMC Sampling with Gibbs Sampler

The goal is to generate 100 approximate samples from the Gibbs distribution of the Ising model for a range of inverse temperatures β. Samples are taken after a sufficiently long number of iterations to ensure the Markov chain is close to its stationary distribution.

### b) Perfect Sampling (Propp–Wilson, Coupling From The Past)

The Propp–Wilson algorithm is implemented to obtain 100 exact samples from the stationary distribution of the Ising model, using the same range of β values. This method guarantees exact sampling through coupling from the past.

_Made by: tuli-pen, Fran7373, jnardosp_
