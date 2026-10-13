# Backout Backprop

This repository asks a simple question: can a neural network learn to predict what happens next without using backpropagation?

[Critical architecture review canvas](./backout-backprop-critical-review.canvas.tsx)

The experiment is motivated by two observations:

1. Animals learn continuously. Each biological connection changes using signals available nearby, and there is no known evidence that the brain performs exact backpropagation.
2. Predicting every pixel in the next video frame wastes effort on details that may not matter. Joint-Embedding Predictive Architectures (JEPAs) instead try to predict a smaller internal description of what will happen [[1](#references), [2](#references), [3](#references)].

This does not mean that the brain is a JEPA or that it cannot use learning signals that resemble backpropagation. Several existing methods already explore more biologically plausible ways to train neural networks [[4](#references), [5](#references), [6](#references), [7](#references)].

## Central question

Can a network learn to predict the next video frame when each connection updates itself using only:

- the recent activity around that connection; and
- a feedback signal saying how wrong the prediction was?

The first goal is to see whether this local learning rule reaches a useful solution at all. A smaller version trained with backpropagation will show us what the architecture can learn under a strong conventional training method, but the local method does not need to match it in the first experiment.

The engineering goal is to remove the expensive backward pass used during training. For many common layers, training with backpropagation costs roughly three times as much as one forward prediction. Training a recurrent network is even more demanding because the system must save or recompute its earlier activity.

Our local method will also have costs: neurons store input histories, and corrections must travel over the active graph. We therefore need to measure the real speed and memory use instead of assuming it will be cheaper.

Later experiments will test larger networks and longer videos. Pruning should also make prediction cheaper over time by removing connections that the network does not need. A failed experiment can still teach us whether the problem was assigning feedback, remembering older events, keeping the network stable, or finding a useful set of connections.

## Deciding which connection deserves credit

Knowing that the final prediction is wrong does not tell us which hidden connection caused the mistake. This is called the **credit-assignment problem**. A single score can say whether the whole network did well, but it cannot say which connection helped or hurt.

The first version follows a simpler rule:

1. For each prediction, every neuron records **each input value it received**, which connection supplied it, the connection's weight and bias at that time, and the neuron's output.
2. When the real next frame arrives, every output neuron receives a signed correction. The correction is positive if its prediction was too low and negative if its prediction was too high. Its size is proportional to the error.
3. Correction travels backward over the **same connections** used for the forward prediction. There is no separate random feedback matrix. Only connections and neurons that were active during that prediction take part.
4. As each correction arrives over a connection, that connection's bias is corrected immediately. Because bias is subtracted in the forward equation, a positive “prediction too low” correction decreases bias.
5. The neuron waits until it has received the expected corrections from all active downstream connections for that prediction. It combines them into one aggregate correction.
6. The neuron then updates the weights of its active incoming connections using the aggregate correction and the exact input values stored for those connections.
7. The aggregate correction is passed backward over those same active incoming connections, and the process continues toward the input.

For a correction $c_{j \leftarrow k}$ arriving at neuron $j$ over connection $j \rightarrow k$, first update that connection's bias:

$$
b_{jk} \leftarrow b_{jk} - \eta_b c_{j \leftarrow k}
$$

After all expected corrections arrive, combine them into $C_j$ and update the incoming weights:

$$
w_{ij} \leftarrow w_{ij} + \eta_w C_j x_i
$$

Here $x_i$ is the exact input that neuron $j$ recorded from connection $i \rightarrow j$ during that prediction. If $x_i$ was zero or the ReLU connection did not activate, that connection is not updated. This is intentionally simpler than backpropagation: it does not calculate a chain of derivatives or use a separate feedback network.

Weight correction keeps a fixed scale and is therefore proportional to the recorded source activation. Connection bias uses a bounded proportional-integral (PI) correction. Each connection keeps a small integral trace of the corrections that have reached it:

$$
I_{ij}[t]
=
\operatorname{clip}\left(
\lambda I_{ij}[t-1]+c_{ij}[t],
-I_{\max},
I_{\max}
\right)
$$

$$
\Delta b_{ij}[t]
=
K_Pc_{ij}[t]+K_II_{ij}[t],
\qquad
b_{ij}\leftarrow b_{ij}-\Delta b_{ij}[t].
$$

The proportional term reacts to the current error. The integral term remembers a persistent error and adds a gradually increasing push in the same direction. Opposite errors unwind that memory instead of resetting it abruptly. The decay $\lambda<1$ and limit $I_{\max}$ prevent old errors from accumulating without bound. The trace is updated only when a nonzero correction actually passes through an active connection; an inactive connection receives neither correction nor reward.

This is still local: $c_{ij}$ is the correction already travelling through connection $i\rightarrow j$. It does not require a derivative, a replayed forward pass, or a separate feedback matrix. Unlike the earlier same-sign rule, it grows linearly with accumulated error rather than doubling exponentially. That earlier rule is retained in [Unsuccessful and inconclusive experiments](UNSUCCESSFUL_EXPERIMENTS.md).

### Ideas removed from the active rule

Positive-only reinforcement, centered reinforcement, signed pruning votes, and connection-pair merging were implemented but did not show useful improvements. They remain documented with equations and measurements in [Unsuccessful and inconclusive experiments](UNSUCCESSFUL_EXPERIMENTS.md), rather than being presented as parts of the current proposal.

This simplicity is also the main risk. The same correction may not be appropriate for every active connection, especially when weights can be negative or several paths cancel each other. The purpose of the first experiment is to find out whether this rough active-path correction is still enough to reach a useful solution. If it fails, later versions can test weight signs, local slopes, eligibility traces, or random feedback one at a time instead of assuming them from the start.

Hinton's Forward-Forward algorithm is another useful comparison. It trains each layer using examples of “good” and “bad” data instead of sending an error backward through the whole network [[9](#references)]. We can test it later, after understanding the simpler rule above.

## Proposed model

### State and connectivity

- Start with a small network in which only some possible connections are active.
- For the current prediction, every neuron keeps the exact input and active path needed when its matching correction arrives. Between pruning events, the implementation keeps only activity counts and pruning votes; it does not keep old activation signatures whose meaning would become stale as weights and biases change.
- Ordinary hidden neurons are stateless between video frames. Remembering an input briefly until its matching correction arrives is training bookkeeping, not temporal neuron state.
- Keep stateful memory neurons as a separate future class. An NMDA-inspired memory neuron may eventually retain and amplify selected activity, but ordinary hidden neurons should not acquire that behavior implicitly.
- The current experiment uses a simple one-frame input delay so it can observe motion before introducing learned memory neurons. It supplies the current and previous frames to the first hidden layer with independent connections.
- Include ordinary forward connections, shortcuts, and a limited number of sideways or backward connections.
- A connection carries its current weight, connection-specific bias, and one bounded PI integral trace. Bias is updated when an individual correction passes through; weight is updated from the receiving neuron's aggregate correction multiplied by the connection's recorded input. Ordinary pruning preserves the traces of surviving connections, discards removed traces, and initializes newly created connections at zero.
- Predict how the next frame will differ from the current frame instead of generating the whole next frame from scratch.

At time $k$, the input delay constructs

$$
z[k]=[x[k],x[k-1]].
$$

Each connection into an ordinary hidden neuron then produces

$$
q_{ij}[k] = \operatorname{ReLU}(z_i[k]w_{ij}[k] - b_{ij}[k]),
$$

and the neuron sums its incoming contributions:

$$
h_j[k] = \sum_i q_{ij}[k].
$$

The first frame of a clip initializes the input delay, and prediction training begins with the second frame. No hidden activation is carried into the next time step. Current-frame and delayed-frame connections have independent weights and gates, so the model can learn how much each source matters instead of being forced to use a fixed temporal difference.

Each connection output is nonnegative. A negative weight or a large positive bias suppresses a connection rather than sending a negative contribution directly. If active inhibition is needed later, it must be represented by a different connection equation; the initial experiment treats inhibition as connection-level suppression. Because a frame delta can be positive or negative, the output uses separate positive and negative channels and subtracts them: $\widehat{\Delta x}=\widehat{\Delta x}^{+}-\widehat{\Delta x}^{-}$.

The whole system is discrete-time by construction:

$$
z[k]=[x[k],x[k-1]], \qquad
h[k] = F(z[k]; w[k], b[k]).
$$

Future memory, recurrent, or sideways neurons will read explicitly stored state from time $k$ to produce time $k+1$; they will not propagate continuously inside one time step. When the real frame at $k+1$ arrives, its correction is matched with the recorded active paths from prediction $k$. The resulting parameter changes are used for later predictions.

Activity counters grow between pruning events. After pruning has used them to find weak connections, they are cleared and a new window begins. Neuron merging instead re-evaluates the current network on a fixed probe set at pruning time.

Purkinje and granule cells are useful inspiration, but looking biologically similar does not prove that two systems learn in the same way. Any biological analogy should lead to a specific idea we can test.

### Removing and creating connections

After a short warm-up that collects enough history for the first decision, prune small groups of connections frequently. Early pruning keeps both forward activity and correction routing manageable. During this exploration period, activate roughly as many new connections as are removed so that an early mistake can be reversed.

Later, it will enter a cleanup phase and remove more connections than it creates. This should make the final network smaller and cheaper to run.

A connection has its own bias, which acts like a gate on that connection. If correction repeatedly pushes the bias so far that the connection almost never activates, the connection serves little purpose and is a pruning candidate. Removing it may not save much arithmetic because it was already inactive, but it removes its lookup, routing, and history overhead.

The larger computational savings may come from merging redundant neurons rather than individual connections. At each merge attempt, the moving-dot experiment re-runs all 288 position-and-direction states through the current network. This avoids deciding from an old path that may have changed since it was recorded. Merging is directional: remove neuron A only when its behavior is contained by survivor B.

- Incoming containment is $|A_{\mathrm{in}}\cap B_{\mathrm{in}}|/|A_{\mathrm{in}}|$; B is allowed to respond to additional inputs.
- Outgoing containment is measured in the same direction; B is allowed to affect additional targets.
- Where A is active, its output must be approximately proportional to B's output. Matching binary activation alone is insufficient.
- Fit B's incoming weights and biases to reconstruct the combined hidden output $A+B$, then fit its outgoing weights and biases to reconstruct both neurons' former contributions.
- Accept the merge only when every affected outgoing target remains within a chosen reconstruction-error tolerance; then remove all incoming and outgoing edges belonging to the redundant neuron.

Bias adjustment alone cannot generally absorb A: changing a bias adds a constant offset while A's contribution varies with its input and weight. Fitting both weight and bias is the smallest affine correction that can represent a varying contribution. The exhaustive reconstruction check decides whether that approximation is actually safe.

If two hidden outputs are proportional, $h_b\approx\alpha h_a$, their outgoing weights can be folded as

$$
w'=\frac{w_a+\alpha w_b}{1+\alpha}, \qquad b'=b_a+b_b.
$$

The code does not assume these approximations are safe: it reconstructs the merged hidden output and every affected outgoing contribution on all 288 states, then rejects the merge if the relative error is too large. This is stricter than matching Jaccard signatures alone.

Pruning is also responsible for exploration. At every pruning event, randomly reactivate a small fraction of inactive connections when nearby neurons have been active and firing. Reset their bias so they can open, give them a small weight, and protect them from immediate re-pruning for one history period. This gives mistakenly suppressed paths a chance to become useful again.

Completely new connections can be chosen randomly at first. Later, we can test whether choosing endpoints near active neurons works better.

Other research has shown that changing a network's wiring during training can help, although strong methods such as RigL use gradients to choose new connections [[10](#references)]. We want to learn whether local signals are enough.

The moving-dot pruning experiment removes connections that were never active during the preceding 1,000-observation window and randomly reactivates `5%` of the removed count near endpoints that were active. Training uses overlapping transitions from continuous ten-frame trajectories. On the old explicit two-frame architecture, across five seeds after 10,000 observations:

- no pruning reaches `74.7%` mean one-step accuracy and `15.9%` rollout accuracy;
- periodic inactive-edge pruning keeps `54.6%` of connections and reaches `72.2%` one-step and `15.9%` rollout accuracy; and
- mean MSE remains nearly identical: `0.01994` without pruning and `0.01992` with pruning.

Pruning therefore compressed that architecture substantially with a modest one-step accuracy cost. A later controlled five-seed test at width 86 exposed an implementation error specific to PI: rebuilding the edge lists reset the integral traces of every surviving connection. At 10,000 observations, matched proportional-only correction reached `89.4%` without pruning and `90.0%` with pruning; PI reached `92.0%` without pruning but only `89.3%` when its traces were reset every 1,000 observations. Preserving surviving traces, deleting only removed traces, and giving reactivated edges zero-valued traces raises pruned PI to `93.6%`. Pruning can therefore coexist with PI; the integral memory must follow the stable identity of each surviving edge.

The neuron-level merge implementation still correctly collapses deliberately duplicated ReLU neurons while preserving their historical outgoing contribution. A successful merge reconstructs several edges rather than preserving their identity, so it still initializes the affected PI traces again; naturally learned merges have not yet occurred in this model.

Directional containment is a better candidate test than symmetric Jaccard similarity, but it does not change the result on this small model. Across five seeds, containment thresholds `0.95`, `0.80`, `0.60`, `0.40`, and `0.30` all produce zero merges at `10%` reconstruction tolerance. In one representative seed, the strongest directed candidate has incoming containment `0.656`, outgoing containment `0.415`, and conditional output cosine similarity `0.716`. Its outgoing behavior is not close to being a subset, and reconstruction rejects it. The safe interpretation remains that this model contains no behavior-preserving neuron redundancy. A larger and deeper model will provide a more meaningful test. Earlier misleading or ineffective pruning and merging attempts are recorded in [Unsuccessful and inconclusive experiments](UNSUCCESSFUL_EXPERIMENTS.md).

Connections are now stored as physical edge lists containing source index, target index, weight, and bias. Pruning consumes and clears the current history, then removes entries from these arrays; reactivation appends only source-target pairs that do not already exist. Forward and correction propagation therefore visit present connections rather than multiplying a dense matrix by a presence mask.

Having fewer connections still does not automatically make a program faster. Scatter operations can lose to highly optimized dense matrix multiplication on a tiny network, even when they perform fewer conceptual operations. We will therefore report both the number of present and active connections and measured running time as the model grows.

### Awake and sleep phases

While “awake,” the network receives real video, predicts the next frame, and saves a small number of recent sequences. During a later “sleep” phase, it receives no new video. It replays saved or imagined sequences while adjusting connections. Earlier work suggests that replay can reduce forgetting [[11](#references)].

An LSTM is probably the wrong memory primitive for this system: its gates are learned, normally by backpropagation through time, whereas this experiment needs storage that works before anything is trained. Start with an untrained, dimension-preserving memory block. For an $n$-value input, it has an $n$-value output and acts as a one-step delay:

$$
y[k]=m[k], \qquad m[k+1]=x[k].
$$

A longer block is a bounded FIFO or ring buffer containing $L$ such vectors. It introduces no learned weights or gates; its explicit cost is $O(Ln)$ memory. This gives the network access to an earlier signal without asking a neuron to discover how to preserve it.

During ordinary forward operation, the block receives a signal from upstream, stores it, and transmits the preceding signal downstream. During replay, it emits the stored sequence in reverse temporal order, from newest to oldest. Biological reverse replay is observed in hippocampal place-cell sequences, although that observation does not establish that the same mechanism will help this learner [[18](#references)].

Reversing time and reversing network direction are different experiments. The first version should reverse only temporal order while using the ordinary forward path. A later bidirectional version can replay toward the neurons that originally produced the signal, but it will need explicit ports and a rule preventing a replayed value from being immediately written back into the same buffer.

This memory block does not solve credit assignment or compress its contents; it simply makes recent state available. Compare no memory, one-step delay, short FIFO, forward replay, and reverse replay. Add sleep only after the awake-only network has learned something useful.

### Weights as temporal memory

Because weights change while the system is running, they may store recent experience as well as long-term structure. This idea is related to **fast weights**, where quickly changing synaptic values act as temporary memory [[16](#references), [17](#references)].

Using one value for both temporary memory and permanent knowledge risks overwriting useful behavior. A later experiment can split each effective weight into slow and fast parts:

$$
w_{ij}[k] = s_{ij}[k] + m_{ij}[k]
$$

$$
m_{ij}[k+1] = \lambda_m m_{ij}[k] + \gamma\,\Delta w_{ij}[k]
$$

The fast part $m$ stores and amplifies recent corrections, then fades according to $\lambda_m$. The slow part $s$ changes only after corrections repeat or are consolidated during sleep. This is computationally similar to giving a synapse a temporary chemical state. The comparison should test whether fast weights improve temporal prediction or merely make the network unstable and forgetful.

### Attention or focus

A separate network could select the important part of each image. However, training that network with backpropagation would mean that the complete system is not backpropagation-free. Begin with images small enough that this focus system is unnecessary. Test focus later as a separate addition.

## First PyTorch test

The smallest implementation is a chain with one input, one hidden neuron, and one output neuron. It uses the connection equation and active-path correction above, but no automatic differentiation, `loss.backward()`, or optimizer.

Create the environment and run it:

```bash
conda env create -f environment.yml
conda activate backout-backprop
python -m experiments.scalar_chain
python -m experiments.two_head_fork
python -m pytest -q
```

The test learns the simple mapping $y=0.8x$. This is not the final video experiment. It checks four basic pieces first:

1. a positive correction increases an active weight;
2. because bias is subtracted, the same correction decreases bias and opens the gate;
3. inactive connections receive no correction; and
4. the full two-connection chain reduces error without PyTorch autograd.

If this test fails, adding images or recurrence will not repair the learning rule.

The second implementation connects the hidden neuron to two output heads. Head 1 learns $y_1=0.8x$ and head 2 learns $y_2=0.2x$. Each head updates independently, while the hidden connection receives their average correction. An explicit test gives the heads equal and opposite corrections and verifies that:

- the head weights move in opposite directions; and
- their corrections cancel exactly at the shared hidden connection.

This tests the intended consensus behavior before replacing scalar input with moving-dot video.

## Minimal experimental ladder

### Stage 0: verify credit assignment

Start with fixed-rate grayscale video containing one object moving at constant velocity and bouncing elastically from the frame boundaries. A one-frame input delay supplies the current and previous frames to the first hidden layer; every hidden layer itself remains stateless. Each later training step follows the intended operating loop:

1. present the current frame $x_t$;
2. predict the frame delta $\widehat{\Delta x_t}$;
3. form the next-frame prediction $\hat{x}_{t+1}=x_t+\widehat{\Delta x_t}$;
4. reveal the real next frame $x_{t+1}$;
5. compute the signed prediction error; and
6. send the correction through the active path and update each active connection from its exact stored input, without automatic differentiation or replaying the forward computation.

Training uses ten consecutive frames from one trajectory, giving eight overlapping prediction-and-correction steps, before restarting the dot from a new random state. This makes history inside each clip temporal while still exposing the model to varied positions and velocities.

First, check whether prediction error goes down during training and eventually levels off. It should not grow without limit or swing wildly.

Next, give the network movement paths that it did not see during training. Test both:

- one-step prediction, where every input frame is real; and
- long prediction, where the network must use its own predicted frame to predict the following frame.

Measure image error, the error in the object's predicted position, the number of active connections, update time, and results across several random starting networks.

The frozen-hidden ablation tests whether backward correction actually teaches the hidden layer or whether the output layer merely learns over fixed random features. Across five seeds after 10,000 observations, the full local learner reaches `74.7%` mean one-step accuracy and residual MSE `0.01994`. Freezing all hidden weights and connection biases while continuing to train the output layer reaches only `0.08%` accuracy and MSE `0.03973`; its rollout is effectively persistence (`7.0%`). A longer single-seed frozen run improves slowly to only `27.1%` one-step accuracy after 100,000 observations, compared with roughly `99%` for the full learner. Hidden correction is therefore essential in this experiment.

Run the current Stage 0 implementation with:

```bash
python -m experiments.moving_dot
```

After 10,000 online observations on continuous ten-frame clips of 6×6 bouncing-dot video, the deterministic PI test run without pruning achieves:

- delta MSE of `0.018293`, compared with `0.055556` for persistence;
- `80.1%` correct next positions; and
- `15.4%` correct positions over an eight-step autoregressive rollout, compared with `7.0%` for persistence.

Across five seeds, the same PI configuration reaches `80.0%` mean one-step accuracy, `17.1%` rollout, and `0.018335` MSE. One-step accuracy has a `2.66` percentage-point standard deviation. Restoring independent current-frame and delayed-frame connections therefore recovers the lost early performance. This does not yet isolate PI's contribution on the restored architecture; a new proportional-only control is required.

The abandoned all-hidden residual-state experiment and its PI comparison are retained in [Unsuccessful and inconclusive experiments](UNSUCCESSFUL_EXPERIMENTS.md). It was not equivalent to input memory: it forced fixed temporal coefficients after nonlinear transformations and added temporal state to every hidden stage.

The saved `moving_dot_final_prediction.png` compares a one-step prediction with the real next frame after advancing through eight real frames. Every prediction uses the real current frame and the input delay retains its predecessor; it does not feed predicted pixels back into the model. Autoregressive rollout remains a separate stress test.

The current defaults are weight correction `0.05`, $K_P=0.0015$, $K_I=0.00015$, $\lambda=0.99$, and $I_{\max}=20$. These values are experimental settings, not yet a general rule for choosing PI gains.

A controlled width sweep on the old explicit two-frame baseline shows that the 48-neuron hidden layer was capacity-limited. Across five seeds after 10,000 observations with ten-frame clips:

- 48 hidden neurons: `79.4%` one-step accuracy, `15.8%` rollout accuracy, `0.01839` MSE, and 13,824 learned parameters;
- 96 hidden neurons: `92.4%`, `18.8%`, `0.01389`, and 27,648 parameters; and
- 192 hidden neurons: `98.2%`, `21.3%`, `0.01065`, and 55,296 parameters.

Mean training time in this small CPU experiment rises from `1.85` to `2.57` and `4.01` seconds respectively. Width strongly improves one-step learning and reduces variation between seeds, but rollout improves much less. More capacity therefore does not solve error accumulation when predictions are fed back into the model.

The input-to-output shortcut experiment was removed after it traded a large loss in one-step accuracy for a smaller rollout improvement. Its results are preserved in [Unsuccessful and inconclusive experiments](UNSUCCESSFUL_EXPERIMENTS.md).

### Stage 0a: sparse longitudinal broadcasters

Dense output-to-hidden recurrence was removed after failing experimentally; its implementation and results are summarized in [Unsuccessful and inconclusive experiments](UNSUCCESSFUL_EXPERIMENTS.md). It should not be confused with sparse longitudinal connectivity.

A different hypothesis is to give a small fraction of neurons very high feed-forward fanout. Most neurons retain sparse local connections, while these broadcaster neurons project across several later layers. Start with many random longitudinal edges, collect local correction and activity evidence, then rapidly deactivate and physically prune paths that are unused or harmful.

This needs a network with at least three hidden layers: in the current one-hidden-layer model, every hidden neuron already connects directly to the output, so “longitudinal” fanout is not a distinct architecture. Compare broadcasters against a control with the same total edge and parameter budget. Otherwise any improvement could come merely from adding connections rather than concentrating them in a few long-range paths.

The first controlled test used the old explicit two-frame input with three 48-neuron hidden layers. Four first-layer broadcasters owned 480 dense long edges to the third hidden layer and output. The equal-budget control distributed exactly 480 long edges randomly across all 48 first-layer neurons. Across five seeds after 10,000 observations:

- adjacent layers alone reach `2.0%` one-step accuracy and `6.9%` rollout;
- uniformly distributed long edges reach `39.8%` and `10.5%`;
- four concentrated broadcasters reach only `5.5%` and `6.5%`; and
- pruning broadcaster edges every 250 observations leaves 207 of 480 edges on average and reaches `4.5%` and `6.5%`.

Long paths clearly help correction cross the three-layer model, but concentrating the same budget in four neurons creates too narrow a representational bottleneck. Rapid inactivity pruning compresses the broadcaster path by `57%` without causing the failure, but it does not rescue it. The more important result is that the adjacent-only deep learner barely trains: the current unsigned mean-correction rule does not yet assign useful credit through three layers. Detailed negative results are retained in [Unsuccessful and inconclusive experiments](UNSUCCESSFUL_EXPERIMENTS.md).

The deep implementation now uses the same one-frame input delay while keeping all three hidden stages stateless. In an initial single-seed 10,000-observation check, adjacent layers alone remain at persistence (`0%` correct next positions), while four broadcasters reach `17.4%`. This is still weak and needs a new controlled multi-seed comparison.

### Stage 0b: increase structural complexity

Before real grayscale video, move beyond one-hot input in controlled steps:

1. predict a translating and bouncing multi-pixel T shape;
2. predict two, five, and then ten independently moving dots, explicitly measuring failures at overlaps and collisions;
3. replace the single hidden layer with at least three hidden layers; and
4. rerun inactive-edge pruning and neuron-level merging as width and depth create more opportunities for redundancy.

These tests preserve known dynamics while adding simultaneous active pixels, ambiguous overlaps, deeper credit assignment, and enough excess capacity for structural merging to become measurable.

### Stage 1: controlled visual dynamics

Start with short real-video clips of one consistent scene, such as a fixed-camera robotic arm or a ball repeatedly bouncing, downsampled to 32×32 grayscale and a low frame rate. Only after that should we add camera motion, changing backgrounds, occlusion, and multiple objects. This gives up the exact position and velocity labels available in generated video, so evaluation will use held-out next-frame error and long rollouts rather than privileged object-state labels.

Test both the next frame and many frames into the future. A model can predict one frame well but quickly drift away from the correct motion when it must reuse its own predictions [[13](#references)].

### Stage 2: latent prediction

Predicting every pixel forces the model to spend effort on texture and noise. It may also learn to produce blurry averages when several futures are possible. Once the basic learning rule works, compare:

1. raw frame-delta prediction;
2. prediction in a fixed random or pretrained latent space; and
3. prediction in a smaller internal representation learned by the local rule, with a safeguard that prevents every input from receiving the same representation.

The third version is closest to JEPA, but it is also the hardest. If both sides are learned together, the network can cheat by giving every frame the same internal representation.

### Stage 3: richer physics

Physion contains simulated scenes with rolling, falling, sliding, and colliding objects. One task asks whether two objects will touch, which tests physical prediction more directly than pixel similarity [[14](#references)].

Real action datasets such as HMDB51 should come later. Camera motion, video compression, editing, and many possible futures make it difficult to tell why a model failed. GoPro was mainly built for removing motion blur, while Human3.6M is more controlled but focuses on human movement.

## What to compare against

Simple comparison methods help us tell whether the network truly learned. Include:

- persistence: $\hat{x}_{t+1}=x_t$;
- a constant-speed prediction when object position is available;
- a network that can change only its final output weights;
- backpropagation/BPTT on the identical forward architecture;
- direct feedback alignment without pruning;
- the local rule with fixed topology;
- the local rule with pruning and random regrowth; and
- deliberately scrambled feedback, which should fail if correct feedback matters.

Report short- and long-term prediction error, object motion error, number of active connections, operation count, measured running time, and neuron-history memory. Repeat each experiment from several random starting points. Storing input history between pruning events could use more space than expected, so it must be included in any efficiency claim.

## Initial success criterion

> On simple videos of moving objects, the local learner predicts far into the future better than “the frame will not change” and “only train the output” methods. Its hidden connections learn reliably from several random starting points, and it keeps most of its accuracy after 80–90% of possible connections have been removed. It does this without automatic differentiation through the network or through time.

The first version does not need to match backpropagation. Showing that local feedback truly trains the hidden network—and measuring where it succeeds, fails, and costs too much—would already be a useful result.

## References

1. LeCun, Y. (2022). [*A Path Towards Autonomous Machine Intelligence*](https://openreview.net/forum?id=BZ5a1r-kVsf).
2. Assran, M. et al. (2023). [*Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*](https://arxiv.org/abs/2301.08243).
3. Bardes, A. et al. (2024). [*V-JEPA: Latent Video Prediction for Visual Representation Learning*](https://openreview.net/forum?id=WFYbBOEOtv).
4. Lillicrap, T. P. et al. (2016). [*Random Synaptic Feedback Weights Support Error Backpropagation for Deep Learning*](https://www.nature.com/articles/ncomms13276).
5. Nøkland, A. (2016). [*Direct Feedback Alignment Provides Learning in Deep Neural Networks*](https://proceedings.neurips.cc/paper/2016/hash/d490d7b4576290fa60eb31b5fc917ad1-Abstract.html).
6. Scellier, B. & Bengio, Y. (2017). [*Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation*](https://doi.org/10.3389/fncom.2017.00024).
7. Bellec, G. et al. (2020). [*A Solution to the Learning Dilemma for Recurrent Networks of Spiking Neurons*](https://www.nature.com/articles/s41467-020-17236-y).
8. Frémaux, N. & Gerstner, W. (2016). [*Neuromodulated Spike-Timing-Dependent Plasticity, and Theory of Three-Factor Learning Rules*](https://doi.org/10.3389/fncir.2015.00085).
9. Hinton, G. (2022). [*The Forward-Forward Algorithm: Some Preliminary Investigations*](https://arxiv.org/abs/2212.13345).
10. Evci, U. et al. (2020). [*Rigging the Lottery: Making All Tickets Winners*](https://proceedings.mlr.press/v119/evci20a.html).
11. Tadros, T. et al. (2022). [*Sleep-Like Unsupervised Replay Reduces Catastrophic Forgetting in Artificial Neural Networks*](https://www.nature.com/articles/s41467-022-34938-7).
12. Hsieh, J.-T. et al. (2018). [*Learning to Decompose and Disentangle Representations for Video Prediction*](https://arxiv.org/abs/1806.04166).
13. Karl, M. et al. (2019). [*On the Difficulty of Learning and Predicting the Long-Term Dynamics of Bouncing Objects*](https://arxiv.org/abs/1907.13494).
14. Bear, D. M. et al. (2021). [*Physion: Evaluating Physical Prediction from Vision in Humans and Machines*](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/d09bf41544a3365a46c9077ebb5e35c3-Abstract-round1.html).
15. Neftci, E. O., Mostafa, H. & Zenke, F. (2019). [*Surrogate Gradient Learning in Spiking Neural Networks*](https://arxiv.org/abs/1901.09948).
16. Ba, J. L. et al. (2016). [*Using Fast Weights to Attend to the Recent Past*](https://papers.nips.cc/paper/2016/hash/9f44e956e3a2b7b5598c625fcc802c36-Abstract.html).
17. Miconi, T., Stanley, K. & Clune, J. (2018). [*Differentiable Plasticity: Training Plastic Neural Networks with Backpropagation*](https://proceedings.mlr.press/v80/miconi18a.html).
18. Foster, D. J. & Wilson, M. A. (2006). [*Reverse Replay of Behavioural Sequences in Hippocampal Place Cells During the Awake State*](https://doi.org/10.1038/nature04587).