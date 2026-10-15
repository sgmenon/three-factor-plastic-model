# Alternative designs

This file preserves mechanisms that produced useful results but were removed
from the active model because a simpler design offered a better engineering
trade-off. These are not failed experiments; they remain candidates if later
tasks expose limitations in the current architecture.

## Ranked negative-feedback tolerance

The dense deep model once gave every active edge a competition score

$$
r_{ij}=s_i d_j q_{ij}
$$

where $s_i$ was the source neuron's fixed outgoing sign, $d_j$ was the desired
change in the receiving neuron, and $q_{ij}$ was the edge's recorded
contribution. Each edge retained a PI history of this score. For every target
neuron, the least-bad $N$ currently negative edges received ordinary
correction while all worse-ranked edges received a stronger bias penalty.
Thus $N$ was a tolerance for poorly scoring active inputs rather than a hard
fan-in limit.

The mechanism produced real gains. In one two-hidden-layer seed after 10,000
observations, $N=0$ reached `2.7%` position accuracy, $N=3$ reached `55.9%`,
$N=6$ reached `66.8%`, $N=8$ reached `63.1%`, $N=10$ reached `60.0%`, and
$N=96$—effectively no severe ranking penalty—reached `40.8%`. The $N=6$
model reached `10.5%` eight-step rollout and `0.021179` MSE while retaining
13,566 of its original 17,856 connections.

The experiment also gave inhibitory neurons a separate tolerance. With
$N_E=6$ and 10% fixed-sign inhibitory neurons, $N_I=6$ reached `64.6%`
one-step accuracy, `14.1%` rollout, and `0.020782` MSE. Raising $N_I$ to 24
made inhibitory neurons more connected but reduced performance to `60.9%`,
`10.4%`, and `0.022020`.

With equal $N_E=N_I=6$, an inhibitory-percentage sweep produced
metric-dependent optima in the same seed: 5% inhibition gave the best
one-step accuracy (`72.1%`), 10% gave the best rollout (`14.1%`), and 2.5%
gave the lowest MSE (`0.019217`). At 30%, performance fell to `54.5%`
one-step and `9.4%` rollout. Because the competition mechanism has now been
removed, these values do not establish the useful inhibitory percentage for
the active structurally local model.

The active model removes this mechanism for three reasons:

1. Ranking negative edges independently for every target adds sorting and
   Python-loop overhead to each correction pass.
2. It stores a second PI scalar for every edge in addition to ordinary bias PI
   state.
3. A fixed Gaussian structural prior performs slightly better without
   dynamically ranking dendrites. At 50% initial connectivity it reaches
   `67.6%` one-step accuracy, `16.4%` rollout, and `0.019925` MSE while
   retaining 7,661 connections. With the ranking code and second PI trace
   physically removed, an isolated run takes 4.77 seconds. The dense $N=6$
   model takes 48.8 seconds and retains 13,566.

The ranked rule also has a weaker biological interpretation than local
structure and activity-dependent plasticity: a neuron must compare and sort
all currently harmful dendrites before updating them. It may still be useful
where fixed locality is inappropriate or a later task benefits substantially
from online competitive rewiring.
