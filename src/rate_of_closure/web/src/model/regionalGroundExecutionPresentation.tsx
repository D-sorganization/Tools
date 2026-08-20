/** Non-executing rendering for regional-ground execution evidence. */

import type { RegionalGroundExecutionPresentationView } from "./regionalGroundExecutionPresentationModel";

const statusText = (view: RegionalGroundExecutionPresentationView): string => {
  const state = view.failure_stage === null
    ? view.status.replace(/_/g, " ")
    : `failed (${view.failure_stage})`;
  return `${state} - ${view.completed} / ${view.total} accepted trials`;
};

/** Render evidence and visibly unavailable controls, with no event handlers. */
export const RegionalGroundExecutionPresentation = ({
  presentation,
}: {
  readonly presentation: RegionalGroundExecutionPresentationView;
}) => (
  <section aria-label="Regional-ground execution evidence">
    <h3>Regional-ground execution</h3>
    <dl>
      <dt>Schema</dt><dd>{presentation.summary.schema_version}</dd>
      <dt>Model</dt><dd>{presentation.summary.model_id}</dd>
      <dt>Model version</dt><dd>{presentation.summary.model_version}</dd>
      <dt>Producer</dt><dd>{presentation.summary.producer}</dd>
      <dt>Producer version</dt><dd>{presentation.summary.producer_version}</dd>
      <dt>Source revision</dt><dd>{presentation.summary.source_revision}</dd>
      <dt>Input digest</dt><dd>{presentation.summary.input_sha256}</dd>
    </dl>
    <p>{presentation.disabled_detail}</p>
    <p aria-live="polite">{statusText(presentation)}</p>
    <button type="button" disabled title={presentation.disabled_detail}>Run study</button>
    <button type="button" disabled title={presentation.disabled_detail}>Cancel study</button>
  </section>
);
