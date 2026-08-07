/** Machine-readable boundaries shared by run export and engineering readouts. */

export const SIMULATION_MODEL_LIMITATIONS = Object.freeze({
  contact_tracking: Object.freeze({
    basis: "tracked_reference_point" as const,
    description:
      "Forced alignment and sampled fixed-ball contact track the clubhead " +
      "reference point, not swept face-mesh contact.",
  }),
  impact_velocity: Object.freeze({
    basis: "clubhead_reference_translation" as const,
    description:
      "The current rigid impact and ball-flight pipeline consumes reference-point " +
      "translation. Shaft-induced contact-point velocity is analyzed separately " +
      "and does not alter flight.",
  }),
});

/** Human-readable statement of the same boundary for impact inspection. */
export const REFERENCE_PIPELINE_LIMITATION =
  "Shaft attribution is a kinematic analysis. The current rigid impact and " +
  "ball-flight pipeline uses tracked-reference translation, not shaft-induced " +
  "contact-point velocity, so the attribution does not alter impact or flight. " +
  "Forced contact aligns the tracked reference point; sampled fixed-ball contact " +
  "also tracks that reference point rather than swept face-mesh contact.";
