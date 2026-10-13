# Unsuccessful and inconclusive experiments

This file records ideas that were implemented and measured but are not part of the current proposed learning rule. Keeping the negative results prevents us from quietly repeating failed experiments or presenting abandoned mechanisms as active design.

## Exponential same-sign correction gain

The first adaptive bias rule stored the sign and length $s_{ij}$ of each connection's current correction streak. Its correction rate doubled after every two consecutive nonzero corrections with the same sign, up to a cap:

$$
\eta_{b,ij}
=
\min\left(
\eta_{b,\max},
\eta_{b,0}\,2^{\lfloor s_{ij}/2\rfloor}
\right),
\qquad
b_{ij}\leftarrow b_{ij}-\eta_{b,ij}c_{ij}.
$$

On the residual-state moving-dot experiment, an initial rate of `0.0015` capped at `0.005` reached `39.8%` one-step accuracy after 10,000 observations and `96.5%` after 100,000. It learned, but the mechanism was needlessly abrupt: two nearly identical correction histories could produce rates differing by a factor of two, an opposite correction erased all memory, and the hard cap—not the rule itself—prevented exponential growth.

The active experiment replaces this with bounded proportional-integral correction. Its memory grows and unwinds gradually. This is a change to a more controlled update rule, not evidence that every form of adaptive correction is harmful.

## Temporal residual state in every hidden layer

An attempted current-frame-only architecture made every hidden stage remember its previous raw activation and expose

$$
h[k]=u[k]+\gamma\left(u[k]-u[k-1]\right).
$$

With $\gamma=1$, this is $2u[k]-u[k-1]$. It was not equivalent to supplying current and previous frames independently: it reused the same weights for both times, fixed their coefficients to $2$ and $-1$, subtracted after the nonlinear connection gates, and repeated the temporal transformation at every hidden depth. Correction updated the current active path but did not assign credit through the stored old state.

With PI correction, the one-hidden-layer model reached `54.1%` one-step accuracy after 10,000 observations in the deterministic seed and `98.0%` after 100,000. Across five seeds at 10,000 observations, PI reached `46.72%` mean one-step accuracy versus `33.24%` for proportional-only correction. PI helped this architecture, but restoring a separate one-frame input delay reached `80.0%` across five seeds at the same checkpoint.

The all-hidden residual mechanism was removed. Ordinary hidden neurons are now stateless, while current and delayed frames have independent first-layer connections. Stateful NMDA-inspired memory neurons remain a separate future experiment rather than an implicit property of every hidden neuron.

## Hard-step connections

A discontinuous connection was tested as

$$
q_i(x_i)=x_iw_i\,\mathbf{1}[x_iw_i-b_i>0].
$$

Across five seeds after 10,000 observations, the dense hard-step model reached `54.2%` mean one-step accuracy, `10.9%` rollout, and `0.02277` MSE. Its dense ReLU control reached `80.8%`, `17.3%`, and `0.01839`. Pruning reduced the hard-step model from 6,912 to about 1,549 edges but lowered one-step accuracy to `48.1%`.

No safe step-neuron merges appeared even with permissive similarity and reconstruction thresholds. The mechanism was both worse and substantially more complicated than ReLU, while a memoryless step is not a realistic spiking neuron. Step support and its special merge implementation were therefore removed from the active code. A future spiking experiment should model time, threshold crossing, and refractory state explicitly rather than reintroducing this simplified switch.

## Reinforcing successful paths

The motivation was to stabilize connections that participated in correct predictions. A stationary target should not earn reward merely because predicting no change was easy, so the first rule rewarded only meaningful target changes:

$$
r =
\mathbb{1}\left[|\Delta x_{\text{target}}| > \delta\right]
\max\left(0, 1 - \frac{|e|}{\epsilon}\right)
$$

Positive-only reward lowered connection bias:

$$
b_{ij} \leftarrow \max\left(0,\ b_{ij} - \eta_r r_{ij}\right)
$$

This has a structural problem: every frequently active connection eventually becomes easier to activate, including passengers that did not cause the successful prediction. It also continues opening connections near convergence, when mean signed correction approaches zero.

A centered variant recorded neuron $j$'s mean meaningful reward $\bar r_j$ during a window of $K$ correction sweeps:

$$
A_{ij} =
\sum_{k:\,i \rightarrow j\ \text{active}}
\left(r_j[k] - \bar r_j\right)
$$

It then applied:

$$
b_{ij} \leftarrow \max\left(0,\ b_{ij} - \eta_r A_{ij}\right)
$$

Across five seeds after 10,000 continuous moving-dot observations:

- no reinforcement reached `74.7%` one-step and `15.9%` rollout accuracy;
- positive-only reinforcement at $10^{-4}$ reached `73.9%` and `15.8%`; and
- centered reinforcement at $10^{-4}$ every 1,000 observations reached `75.0%` and `15.4%`.

The centered rule's `0.3` percentage-point one-step gain was inconsistent across checkpoints, rollout became slightly worse, and rates of $3\times10^{-4}$ or greater degraded both metrics. Reinforcement remains implemented for reproducibility but is disabled by default.

## Signed pruning votes

The experiment recorded a non-forward score for each active connection. Positive aggregate correction incremented the score and negative aggregate correction decremented it. The score affected only pruning, not prediction.

This did not measure usefulness. Negative correction often means that a useful connection currently contributes too much, not that the connection should be removed.

Across five seeds with pruning every 1,000 observations:

- thresholds of `-50` and `-25` destroyed learning;
- a `-100` threshold reduced one-step accuracy from `72.2%` to `50.1%`; and
- a `-200` threshold removed only `1.4` scored connections on average and added no useful compression.

Signed-score pruning remains optional in the experiment code but is disabled by default.

## Connection-pair merging

The first merging rule compared historical contributions from pairs of connections entering the same neuron. Pairs needed at least `95%` cosine similarity, and a fitted survivor needed relative reconstruction error no greater than `10%`.

It found only `7.2` merge opportunities on average across five seeds and 10,000 observations. Final connection presence and prediction quality were effectively unchanged. Exact duplicate connections merge correctly, but the small model contains few safely interchangeable connection pairs.

An initial whole-neuron rule used symmetric Jaccard similarity. That was also the wrong geometry: a removable neuron's behavior may be a subset of a more general survivor. The current experiment uses directional incoming and outgoing containment, conditional output proportionality, and exhaustive reconstruction through the survivor.

## Input-to-output shortcuts

An optional direct input-to-output ReLU path was tested as a possible source of additional nonlinearity. It improved eight-step rollout but substantially damaged clean one-step position accuracy. Across five seeds at 192 hidden neurons, rollout increased from `21.3%` to `26.6%`, while one-step accuracy fell from `98.2%` to `75.3%`. Similar behavior appeared at widths 48 and 96.

The shortcut added nonlinear basis terms, but it did not provide sigmoid-like bounding or sharpening. It also added 5,184 connections and forced the direct and hidden paths to learn from the same correction rate. The implementation was removed rather than retained as part of the main architecture.

## Dense positive recurrence

An output-to-hidden recurrent layer fed the 72 output channels back after one time-step delay. The delayed value was the output actually experienced on the preceding frame, and recurrent connections received the matching stored input and local hidden correction without backpropagation through time.

Across five seeds at 48 hidden neurons and 10,000 observations, fixed random recurrence collapsed to `0%` one-step accuracy and persistence-level rollout. Locally updated recurrence reached `72.7%` warm one-step accuracy and `16.0%` rollout, compared with `77.4%` and `16.1%` for the non-recurrent control. Pruning recurrent paths every 250 observations removed all 3,456 learned recurrent edges and recovered to `75.0%` and `15.8%`.

The current ReLU connection can only add nonnegative activity. Dense recurrent loops therefore amplify the hidden state rather than providing stabilizing inhibition. Learning suppresses the recurrent path, while fixed recurrence can overwhelm the output gates. The implementation was removed. Sparse feed-forward longitudinal broadcasters remain a separate hypothesis.

## Concentrated longitudinal broadcasters

A three-hidden-layer model tested whether a small population with high feed-forward fanout carries correction and useful features across depth. Four of 48 first-layer neurons received 480 total long edges to the third hidden layer and output. An equal-budget control distributed exactly 480 long edges randomly across all 48 first-layer neurons.

Across five seeds after 10,000 observations, the adjacent-only model reached `2.0%` one-step accuracy, uniformly distributed long edges reached `39.8%`, and concentrated broadcasters reached only `5.5%`. Their eight-step rollout scores were `6.9%`, `10.5%`, and `6.5%`. Pruning broadcaster edges every 250 observations retained 207 of 480 edges on average but reduced one-step accuracy to `4.5%`.

Long-range paths help this deep local learner, but concentrating them in four neurons is much worse than spreading the same edge budget. Four random features cannot carry enough position-specific information for the moving-dot task. The adjacent-only failure also shows that the current mean active-path correction does not assign useful credit through three hidden layers. The implementation remains available as an ablation, but concentrated broadcasters are not part of the active proposal.

## Apparent convergence gain from shuffled transitions

An early moving-dot version sampled an independent three-frame transition for every update. In that setup, periodic pruning improved mean one-step accuracy from `70.0%` to `72.6%`.

The samples were valid transitions but not continuous video. After training was corrected to use overlapping transitions from ten-frame trajectories, the convergence advantage disappeared. Inactive-edge pruning still compressed the network, but the earlier acceleration should not be treated as evidence.
