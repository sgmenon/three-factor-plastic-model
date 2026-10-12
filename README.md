# Backout Backprop

This repository asks a simple question: can a neural network learn to predict what happens next without using backpropagation?

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

## Two learning problems

### 1. Neurons that fire in discrete steps

A hard threshold jumps directly from off to on. There is no useful derivative at the point where it jumps, so ordinary backpropagation cannot pass a gradient through that event. Spiking neural networks usually work around this by pretending that the jump has a smooth derivative during the backward pass [[15](#references)].

A step function by itself is not a realistic biological neuron. A real neuron builds up electrical charge over time, fires a spike after crossing a threshold, needs time before firing again, and is affected by noise. We eventually want to compare neuron models that include these changes over time.

A derivative is not required by the simple correction rule below. A smooth activation is still useful as an engineering control because a small weight change usually produces a small, visible change in the neuron's output. With a hard threshold, many weight changes do nothing until the neuron crosses the threshold, at which point its output changes suddenly. That can make it harder to tell whether a correction helped.

Success with smooth neurons would not prove that hard-threshold neurons will learn. It would only show that the active-path correction works when neuron outputs change gradually. This separates two questions:

1. Can local feedback train a network without end-to-end backpropagation?
2. Can the same idea work when neuron outputs are discrete and no real derivative exists?

The first question is the main experiment. The second is an optional, harder follow-up motivated by biology. If we pursue it, use the same derivative-free active-path correction rule in both cases and change only the neuron activation:

1. smooth neurons whose outputs change gradually; and
2. hard-threshold or spiking neurons whose outputs are discrete.

This direct comparison will tell us whether discontinuous firing itself prevents the simple correction rule from learning. Surrogate derivatives and established three-factor spiking rules can be added later as reference methods [[8](#references), [15](#references)], but they should not be part of the initial hypothesis.

### 2. Deciding which connection deserves credit

Knowing that the final prediction is wrong does not tell us which hidden connection caused the mistake. This is called the **credit-assignment problem**. A single score can say whether the whole network did well, but it cannot say which connection helped or hurt.

The first version follows a simpler rule:

1. For each prediction, every neuron records **each input value it received**, which connection supplied it, the connection's weight and bias at that time, and the neuron's output.
2. When the real next frame arrives, every output neuron receives a signed correction. The correction is positive if its prediction was too low and negative if its prediction was too high. Its size is proportional to the error.
3. Correction travels backward over the **same connections** used for the forward prediction. There is no separate random feedback matrix. Only connections and neurons that were active during that prediction take part.
4. As each correction arrives over a connection, that connection's bias is corrected immediately.
5. The neuron waits until it has received the expected corrections from all active downstream connections for that prediction. It combines them into one aggregate correction.
6. The neuron then updates the weights of its active incoming connections using the aggregate correction and the exact input values stored for those connections.
7. The aggregate correction is passed backward over those same active incoming connections, and the process continues toward the input.

For a correction $c_{j \leftarrow k}$ arriving at neuron $j$ over connection $j \rightarrow k$, first update that connection's bias:

$$
b_{jk} \leftarrow b_{jk} + \eta_b c_{j \leftarrow k}
$$

After all expected corrections arrive, combine them into $C_j$ and update the incoming weights:

$$
w_{ij} \leftarrow w_{ij} + \eta_w C_j x_i
$$

Here $x_i$ is the exact input that neuron $j$ recorded from connection $i \rightarrow j$ during that prediction. If $x_i$ was zero or the receiving neuron did not activate, that connection is not updated. For a smooth activation, “active” means that its output exceeded a small chosen threshold; for a spiking neuron, it means that it fired. This is intentionally simpler than backpropagation: it does not calculate a chain of derivatives or use a separate feedback network.

This simplicity is also the main risk. The same correction may not be appropriate for every active connection, especially when weights can be negative or several paths cancel each other. The purpose of the first experiment is to find out whether this rough active-path correction is still enough to reach a useful solution. If it fails, later versions can test weight signs, local slopes, eligibility traces, or random feedback one at a time instead of assuming them from the start.

Hinton's Forward-Forward algorithm is another useful comparison. It trains each layer using examples of “good” and “bad” data instead of sending an error backward through the whole network [[9](#references)]. We can test it later, after understanding the simpler rule above.

## Proposed model

### State and connectivity

- Start with a small network in which only some possible connections are active.
- Every neuron keeps a history of each input value it received, the connection that supplied it, the connection's weight and bias, its summed input, and its output. Each record is tagged with the frame or prediction that created it so the arriving correction can use the matching inputs.
- Each neuron also has a running state that carries some information from one frame to the next. A simple first version is $h_t = \alpha h_{t-1} + (1-\alpha)\,\text{new activation}$, where $\alpha$ controls how quickly old activity fades.
- Include ordinary forward connections, shortcuts, and a limited number of sideways or backward connections.
- A connection stores no learning history or eligibility state. It identifies the two neurons it joins and carries only its current weight and connection-specific bias. The neurons own the history. Bias is updated when an individual correction passes through; weight is updated after the receiving neuron has aggregated its corrections.
- Predict how the next frame will differ from the current frame instead of generating the whole next frame from scratch.

For input $x_i[k]$ at discrete time $k$, connection $i \rightarrow j$ produces:

$$
q_{ij}[k] = \operatorname{ReLU}(x_i[k]w_{ij}[k] - b_{ij}[k])
$$

Neuron $j$ adds the outputs of its active incoming connections:

$$
h_j[k] = \sum_i q_{ij}[k]
$$

This equation makes every connection output nonnegative. A negative weight or a large positive bias suppresses a connection rather than sending a negative value to the next neuron. If active inhibition is needed later, it must be represented by a different connection equation; the initial experiment treats inhibition as connection-level suppression. Because a frame delta can be positive or negative, the output should use separate positive and negative channels and subtract them: $\widehat{\Delta x}=\widehat{\Delta x}^{+}-\widehat{\Delta x}^{-}$.

The whole system is discrete-time by construction:

$$
h[k+1] = F(h[k], x[k]; w[k], b[k])
$$

Recurrent and sideways connections read the stored state from time $k$ to produce time $k+1$; they do not propagate continuously inside one time step. When the real frame at $k+1$ arrives, its correction is matched with the recorded active paths from prediction $k$. The resulting parameter changes are used for later predictions.

Input history grows between pruning events. After a pruning event has used that history to find weak and redundant connections, the old history is cleared and the neurons begin recording a new period. The pruning interval therefore sets the maximum history size.

Purkinje and granule cells are useful inspiration, but looking biologically similar does not prove that two systems learn in the same way. Any biological analogy should lead to a specific idea we can test.

### Removing and creating connections

After a short warm-up that collects enough history for the first decision, prune small groups of connections frequently. Early pruning keeps both forward activity and correction routing manageable. During this exploration period, activate roughly as many new connections as are removed so that an early mistake can be reversed.

Later, it will enter a cleanup phase and remove more connections than it creates. This should make the final network smaller and cheaper to run.

A connection has its own bias, which acts like a gate on that connection. If correction repeatedly pushes the bias so far that the connection almost never activates, the connection serves little purpose and is a pruning candidate. Removing it may not save much arithmetic because it was already inactive, but it removes its lookup, routing, and history overhead.

The larger computational savings should come from merging redundant active connections. At pruning time, compare the histories of connections entering the same neuron:

- If two source neurons almost always fire together and their connections have similar effects, keep one connection and remove the other.
- Increase the surviving weight enough to preserve their combined contribution. If the two source activations are nearly equal, a first approximation is $w_{\text{keep}} \leftarrow w_{\text{keep}} + w_{\text{remove}}$. If one activation is usually a fixed proportion of the other, scale the transferred weight by that proportion.
- Temporarily removing the candidate and checking whether prediction error changes can prevent useful connections from being merged accidentally.

After pruning finishes, clear the history used for that decision. Comparing every possible pair can be expensive for a neuron with many inputs, so the first version can compare only active connections or use the full comparison during infrequent pruning events.

Pruning is also responsible for exploration. At every pruning event, randomly reactivate a small fraction of inactive connections when nearby neurons have been active and firing. Reset their bias so they can open, give them a small weight, and protect them from immediate re-pruning for one history period. This gives mistakenly suppressed paths a chance to become useful again.

Completely new connections can be chosen randomly at first. Later, we can test whether choosing endpoints near active neurons works better.

Other research has shown that changing a network's wiring during training can help, although strong methods such as RigL use gradients to choose new connections [[10](#references)]. We want to learn whether local signals are enough.

Having fewer connections does not automatically make a program faster. The software or hardware must actually skip missing connections. We will therefore report both the number of operations the sparse network should need and its measured running time.

### Awake and sleep phases

While “awake,” the network receives real video, predicts the next frame, and saves a small number of recent sequences. During a later “sleep” phase, it receives no new video. It replays saved or imagined sequences while adjusting connections. Earlier work suggests that replay can reduce forgetting [[11](#references)].

We should not begin with an LSTM memory block because LSTMs are normally trained by backpropagation through time. Start with simple fading neuron states and saved sequences. Add sleep only after the awake-only network has learned something useful.

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


## Minimal experimental ladder

### Stage 0: verify credit assignment

Start with fixed-rate grayscale video containing one object moving at constant velocity and bouncing elastically from the frame boundaries. Each training step follows the intended operating loop:

1. present the current frame $x_t$;
2. predict the frame delta $\widehat{\Delta x_t}$;
3. form the next-frame prediction $\hat{x}_{t+1}=x_t+\widehat{\Delta x_t}$;
4. reveal the real next frame $x_{t+1}$;
5. compute the signed prediction error; and
6. send the correction through the active path and update each active connection from its exact stored input, without automatic differentiation or replaying the forward computation.

First, check whether prediction error goes down during training and eventually levels off. It should not grow without limit or swing wildly.

Next, give the network movement paths that it did not see during training. Test both:

- one-step prediction, where every input frame is real; and
- long prediction, where the network must use its own predicted frame to predict the following frame.

Measure image error, the error in the object's predicted position, the number of active connections, update time, and results across several random starting networks.

We also need to prove that the hidden connections are learning. Compare the full network with a version in which hidden weights are frozen and only the final output weights can change. Also compare it with an untrained network and a simple guess that the next frame will equal the current one.

### Stage 0b: test discontinuous units

Repeat Stage 0 with hard-threshold or spiking neurons while keeping the same active-path correction rule. Compare this directly with the smooth-neuron network. Measure how quickly each one learns, its final error, how often its neurons fire, how expensive each update is, and whether small changes near the firing threshold make training unstable.

### Stage 1: controlled visual dynamics

Generate small 32×32 videos containing one or two moving shapes. Add wall bounces, object collisions, and objects passing in front of each other. Generated video is cheap, and we know the true position and speed of every object. Moving MNIST and Bouncing Balls are established examples [[12](#references), [13](#references)].

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