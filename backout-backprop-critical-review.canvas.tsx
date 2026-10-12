import {
  Callout,
  Card,
  CardBody,
  CardHeader,
  Code,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Table,
  Text,
  useHostTheme,
} from "cursor/canvas";

const findings = [
  [
    "High",
    "Nonnegative connection outputs limit what one channel can represent",
    "q = ReLU(x·w − b) suppresses negative contributions instead of transmitting inhibition. A raw frame delta still needs both positive and negative values.",
    "Use paired ON/OFF output channels and test whether hidden monotonicity limits collision prediction.",
  ],
  [
    "High",
    "The correction is a consensus signal, not a gradient",
    "Ignoring downstream magnitudes prevents one head from dominating, but the update has the correct sign only under constraints such as nonnegative active weights and monotone units.",
    "Map convergence for positive, suppressed-negative, and signed-active edges in a two-weight chain.",
  ],
  [
    "Defined",
    "Recurrence is discrete by construction",
    "The state update h[k+1] = F(h[k], x[k]) makes recurrent and lateral edges ordinary delayed state transitions, not continuous same-step loops.",
    "Tag correction with prediction k and verify that it updates only later predictions.",
  ],
  [
    "High",
    "Neuron-owned full history scales as O(K × E)",
    "Recording every edge input for K frames still stores one value per edge per frame, even though the data lives on neurons rather than connections.",
    "Store each source activation once per frame and let edge records reference it.",
  ],
  [
    "High",
    "Multi-output errors can cancel",
    "A hidden neuron may help one output and hurt another. A scalar aggregate can become zero even when large, useful corrections exist.",
    "Test sum, mean, sign vote, and patch-local correction on a two-output toy problem.",
  ],
  [
    "Requirement",
    "Inactive gates rely on pruning-time exploration",
    "The proposed neighbor-triggered random reactivation is the only way a mistakenly suppressed connection can recover.",
    "Ablate reactivation and measure recovery after intentionally bad pruning.",
  ],
  [
    "Resolved condition",
    "Frozen prediction snapshots remove most message-order dependence",
    "Additive bias updates commute if the active mask from prediction k is frozen and all changes apply only to k+1.",
    "Shuffle message order and verify identical results within floating-point tolerance.",
  ],
  [
    "Medium",
    "Redundancy merging is valid only on the measured distribution",
    "Two neurons that co-fire in one set of scenes may separate under a new motion, camera, or object.",
    "Validate each merge on held-out scenes and retain a rollback window.",
  ],
  [
    "Measurement",
    "Sparse graphs may not be faster on current hardware",
    "Irregular message passing can lose to dense matrix operations despite using fewer arithmetic operations.",
    "Report edge operations, bytes moved, and wall-clock latency separately.",
  ],
];

const modifications = [
  [
    "Nonnegative forward weights",
    "For monotone activations, active-path corrections are more likely to have the correct sign.",
    "Use connection bias for suppression and paired output channels for signed frame deltas.",
  ],
  [
    "Sign-only downstream information",
    "Multiplying correction by sign(weight) fixes the most obvious wrong-direction case without calculating derivatives.",
    "Keep this only as a later control if active signed connections are introduced.",
  ],
  [
    "Normalized aggregation",
    "Prevents correction magnitude from depending mainly on fan-out.",
    "Compare sum, mean, and square-root fan-out normalization.",
  ],
  [
    "Frozen prediction snapshots",
    "Removes asynchronous update ambiguity and makes each result reproducible.",
    "Store prediction ID, active mask, input references, and expected correction count.",
  ],
  [
    "Patch-local outputs",
    "Reduces cancellation between unrelated pixel errors.",
    "Route each hidden region to a limited spatial output neighborhood first.",
  ],
];

function SeverityPill({ level }: { level: string }) {
  return <Pill active={level === "Critical"} size="sm">{level}</Pill>;
}

export default function BackoutBackpropCriticalReview() {
  const theme = useHostTheme();

  return (
    <Stack gap={20} style={{ padding: 24, maxWidth: 1180, margin: "0 auto" }}>
      <Stack gap={8}>
        <Row gap={10} align="center" wrap>
          <H1>Backout Backprop: critical architecture review</H1>
          <Pill size="sm">README design · 12 Oct 2026</Pill>
        </Row>
        <Text tone="secondary">
          Mathematical and engineering review of the active-path correction,
          connection-specific gating, history-driven pruning, and continual video learning proposal.
        </Text>
      </Stack>

      <Callout tone="warning" title="Verdict">
        Worth testing. The strongest idea is not simply “no backprop”; it is the combination of
        continual self-supervised video targets, connection-level gates, active-path correction,
        and topology compression. The correction is best understood as a downstream consensus
        signal rather than an estimate of the gradient. That can reduce interference between heads,
        but its direction is reliable only for a restricted forward system.
      </Callout>

      <Grid columns="1.15fr 0.85fr" gap={16} align="stretch">
        <Card size="lg">
          <CardHeader trailing={<Pill active size="sm">Decisive test</Pill>}>
            Two-layer linear counterexample
          </CardHeader>
          <CardBody>
            <Stack gap={10}>
              <Text>
                Consider one input and one output with prediction{" "}
                <Code>ŷ = w₂w₁x</Code> and error <Code>e = y − ŷ</Code>.
              </Text>
              <Text>The proposed updates are:</Text>
              <div
                style={{
                  background: theme.fill.tertiary,
                  color: theme.text.primary,
                  border: `1px solid ${theme.stroke.tertiary}`,
                  borderRadius: 6,
                  padding: 12,
                  fontFamily: "monospace",
                  lineHeight: 1.7,
                }}
              >
                Δw₂ = η e (w₁x)
                <br />
                Δw₁ = η e x
              </div>
              <Text>
                The output update is the ordinary local delta rule. The hidden update is missing
                the downstream factor <Code>w₂</Code>. For small η, the loss decreases only when
                the combined direction is favorable. If <Code>w₂ &lt; −w₁²</Code>, the first-order
                loss change is positive: training moves uphill.
              </Text>
              <Text>
                With the now-defined edge rule <Code>q = ReLU(xw − b)</Code>, this counterexample
                is excluded when source activity and bias are nonnegative: a negative-weight edge
                cannot remain active. It returns if negative biases permit active negative slopes,
                or if later versions transmit signed inhibition.
              </Text>
              <Text weight="semibold">
                The decisive experiment is therefore a boundary test: nonnegative active edges,
                suppressive negative edges, and genuinely signed edges.
              </Text>
            </Stack>
          </CardBody>
        </Card>

        <Stack gap={12}>
          <H2>What is genuinely promising</H2>
          <Callout tone="success" title="Natural online target">
            Video reveals its own next-frame target. The system can receive corrective data during
            normal operation without labels or a separate training session.
          </Callout>
          <Callout tone="info" title="Bias as a structural gate">
            A connection-specific bias that learns quickly while weights change only after aggregate
            feedback creates two learning timescales: routing first, representation second.
          </Callout>
          <Callout tone="neutral" title="Function-preserving merge">
            If two source activations satisfy h₂ ≈ r·h₁, replacing their two edges with one edge
            weighted w₁ + r·w₂ approximately preserves the receiving neuron’s input. That is a
            concrete, testable compression rule.
          </Callout>
        </Stack>
      </Grid>

      <Stack gap={10}>
        <H2>Scientific findings</H2>
        <Table
          headers={["Priority", "Finding", "Why it matters", "Cheapest decisive test"]}
          rows={findings.map(([priority, ...rest]) => [
            <SeverityPill level={priority} />,
            ...rest,
          ])}
          rowTone={["warning", "warning", "info", "warning", "warning", "info", "info", "neutral", "info"]}
          striped
          stickyHeader
        />
      </Stack>

      <Divider />

      <Stack gap={12}>
        <H2>Novelty worth preserving</H2>
        <Grid columns={2} gap={16}>
          <Stack gap={6}>
            <H3>Two-timescale plasticity</H3>
            <Text tone="secondary">
              Per-message bias changes can select routes quickly; aggregate weight changes can learn
              more slowly. Treat this as a core hypothesis and measure the two timescales separately.
            </Text>
          </Stack>
          <Stack gap={6}>
            <H3>Computation follows activity</H3>
            <Text tone="secondary">
              Only the graph that participated in a prediction receives correction. This makes
              learning cost proportional to active structure and naturally couples learning with sparsity.
            </Text>
          </Stack>
          <Stack gap={6}>
            <H3>Learning and compression share evidence</H3>
            <Text tone="secondary">
              The same activation history used to route correction can identify redundant pathways.
              This could turn pruning from a separate optimization stage into part of continual learning.
            </Text>
          </Stack>
          <Stack gap={6}>
            <H3>Weights can become short-term state</H3>
            <Text tone="secondary">
              Runtime updates can encode recent context directly in synapses. A fast component that
              decays and a slow component consolidated during sleep would test this without requiring
              an LSTM or GRU.
            </Text>
          </Stack>
        </Grid>
      </Stack>

      <Stack gap={10}>
        <H2>Minimal modifications that preserve the idea</H2>
        <Table
          headers={["Modification", "Benefit", "Experiment"]}
          rows={modifications}
          striped
        />
      </Stack>

      <Card collapsible defaultOpen size="lg">
        <CardHeader>Recommended experiment order</CardHeader>
        <CardBody>
          <Stack gap={12}>
            <Text><Text weight="semibold">1. Scalar chain.</Text> Map convergence over initial w₁ and w₂. This should take minutes and can falsify the unrestricted rule.</Text>
            <Text><Text weight="semibold">2. Two-path cancellation.</Text> One positive and one negative path to the same output. Test whether aggregate correction strengthens the wrong path.</Text>
            <Text><Text weight="semibold">3. Two outputs.</Text> Give one hidden neuron conflicting output errors. Compare aggregation rules.</Text>
            <Text><Text weight="semibold">4. Two-node recurrent cycle.</Text> Define exactly when correction stops and verify that message processing order does not change the result.</Text>
            <Text><Text weight="semibold">5. Moving dot video.</Text> Only after the first four pass, test next-frame and rollout prediction.</Text>
            <Text><Text weight="semibold">6. Add gating and pruning separately.</Text> Establish learning first; then test whether bias gates and redundancy merging improve cost without destroying accuracy.</Text>
          </Stack>
        </CardBody>
      </Card>

      <Callout tone="info" title="A strong first paper-sized claim">
        A derivative-free, active-path correction rule can train a monotone recurrent predictor
        online; connection-specific gates reduce active computation; and history-based merges
        compress the graph while approximately preserving its function. This is narrower than
        “replace backprop,” but it is precise, surprising if true, and experimentally defensible.
      </Callout>

      <Text size="small" tone="tertiary">
        Source: current project README and equations. This review evaluates the proposed rule on its
        own terms; comparisons to feedback alignment, e-prop, and dynamic sparse training are controls,
        not requirements.
      </Text>
    </Stack>
  );
}
