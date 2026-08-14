# Impact-to-Flight Solution Families

## Scope

`impact-solution-request/v1` maps desired canonical ball-flight objectives onto
bounded club-delivery parameters. `impact-solution-result/v1` preserves
multiple distinct delivery regions instead of implying that one inverse answer
is unique. The Python runtime supplies a concrete centered driver/iron adapter;
TypeScript supplies the same persistence and interchange contract.

The adapter reuses the canonical chain:

1. `DeliveryParameters` in the app frame (`x` target, `y` up, `z` right);
2. centered rigid-body impact at the declared first-contact event time;
3. the app-to-flight pure rotation (`x` forward, `y` left, `z` up);
4. post-impact launch-condition derivation; and
5. the requested registered literature flight model.

The result target frame is `x` downrange, `y` up and `z` right. Launch
direction and carry offline are positive right under the `app_native`
convention. All request and result units are canonical SI except angles in
degrees and spin in RPM, as declared by the metric catalog.

## Supported Delivery Variables

The v1 centered adapter accepts only:

- clubhead speed (`clubhead_speed_mps`, m/s);
- club path (`club_path_deg`, deg);
- face angle (`face_angle_deg`, deg);
- attack angle (`attack_angle_deg`, deg); and
- dynamic loft (`dynamic_loft_deg`, deg).

Unlisted variables fail request validation. In particular, the contract does
not pretend to support shaft flex, body torques, lie delivery, off-center
contact, turf, head deformation or equipment optimization.

## Family Construction

The existing `inverse-flight-result/v1` bounded search remains authoritative
for sampling, scoring and feasibility. Returned feasible candidates are
visited in rank order. A candidate joins the first family whose representative
lies within the requested Euclidean radius after every delivery parameter is
normalized by its declared bound span. Otherwise it starts a new family until
the requested family count is reached.

Each family retains its ranked members, representative launch state,
launch-objective and flight-objective residuals, per-parameter observed
intervals, Pearson correlations where at least two members exist, and bounded
finite-difference sensitivities at the representative. Sensitivity evaluations
are diagnostic probes and do not alter the inverse solver's attempted-count
contract. A sensitivity is omitted when neither bounded neighboring probe
completes; no zero derivative is fabricated for an unavailable probe.

Every original inverse sample appears exactly once: either as a family member
or a rejected candidate. Rejections distinguish an objective miss, no impact,
model unavailability, a forward failure, the inverse candidate return budget,
and the requested family count/radius.

## Model Availability and Limitations

The model manifest names the impact and flight models and reports availability
before a solution is interpreted. Unknown flight models return
`model_unavailable`; a delivery with nonpositive velocity along the delivered
face normal returns `no_impact`. Neither case receives fabricated launch or
flight values.

The centered driver and iron use documented representative mass/MOI defaults
inside this adapter. They are not manufacturer-specific fits or equipment
certifications. The rigid-body model is deterministic and appropriate for
software integration and comparative studies, but it does not establish a
player's feasible swing capabilities. Continuous/global optimality is not
claimed: families summarize the finite deterministic sample requested by the
caller.

## Parity and Extension

`impact_solution_families_golden_v1.json` is parsed and serialized by both
Python and TypeScript. New physics should enter through another rich forward
evaluator implementing the same model manifest and evaluation protocol. The
family solver must not reach into a specific impact, wind, ground or swing
implementation.
