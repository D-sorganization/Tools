import type {
  RegionalExecutionEventReadback,
  RegionalExecutionTransitionReadback,
} from "../model/regionalExecutionReadback";

const MAX_VISIBLE_LEDGER_ROWS = 256;

const vector = (value: readonly number[]): string =>
  `(${value.map((item) => item.toFixed(6)).join(", ")})`;

const regionSurface = (regionId: string | null, surfaceId: string): string =>
  `${regionId ?? "base"} / ${surfaceId}`;

function LedgerSummary(props: { readonly total: number }) {
  const shown = Math.min(props.total, MAX_VISIBLE_LEDGER_ROWS);
  return <span>{shown === props.total
    ? `${props.total} validated row(s)`
    : `showing first ${shown} of ${props.total} validated rows`}</span>;
}

export function RegionalExecutionLedgerTables(props: {
  readonly events: readonly RegionalExecutionEventReadback[];
  readonly transitions: readonly RegionalExecutionTransitionReadback[];
}) {
  return <div className="mt-4 grid gap-3">
    <details>
      <summary aria-label="Toggle ground execution events"
        className="cursor-pointer font-medium">
        Events · <LedgerSummary total={props.events.length} />
      </summary>
      <div className="mt-2 overflow-x-auto">
        <table aria-label="Ground execution events" className="min-w-max text-left text-xs">
          <thead><tr>
            {[
              "Seq", "Event", "t (s)", "Position (m)", "v before (m/s)",
              "v after (m/s)", "omega before (rad/s)", "omega after (rad/s)", "Frame",
            ].map((heading) => <th className="px-2 py-1" key={heading}>{heading}</th>)}
          </tr></thead>
          <tbody>{props.events.slice(0, MAX_VISIBLE_LEDGER_ROWS).map((event) => <tr
            className="border-t border-slate-700" key={event.sequence}>
            <td className="px-2 py-1">{event.sequence}</td>
            <td className="px-2 py-1">{event.eventType}</td>
            <td className="px-2 py-1">{event.timeS.toFixed(6)}</td>
            <td className="px-2 py-1">{vector(event.positionM)}</td>
            <td className="px-2 py-1">{vector(event.velocityBeforeMps)}</td>
            <td className="px-2 py-1">{vector(event.velocityAfterMps)}</td>
            <td className="px-2 py-1">{vector(event.angularVelocityBeforeRadS)}</td>
            <td className="px-2 py-1">{vector(event.angularVelocityAfterRadS)}</td>
            <td className="px-2 py-1">{event.frame}</td>
          </tr>)}</tbody>
        </table>
      </div>
    </details>
    <details>
      <summary aria-label="Toggle regional surface transitions"
        className="cursor-pointer font-medium">
        Transitions · <LedgerSummary total={props.transitions.length} />
      </summary>
      <div className="mt-2 overflow-x-auto">
        <table aria-label="Regional surface transitions"
          className="min-w-max text-left text-xs">
          <thead><tr>
            {[
              "Event seq", "t (s)", "Position (m)",
              "From region / surface", "To region / surface",
            ].map((heading) => <th className="px-2 py-1" key={heading}>{heading}</th>)}
          </tr></thead>
          <tbody>{props.transitions.slice(0, MAX_VISIBLE_LEDGER_ROWS).map((item) => <tr
            className="border-t border-slate-700" key={item.eventSequence}>
            <td className="px-2 py-1">{item.eventSequence}</td>
            <td className="px-2 py-1">{item.timeS.toFixed(6)}</td>
            <td className="px-2 py-1">{vector(item.positionM)}</td>
            <td className="px-2 py-1">
              {regionSurface(item.fromRegionId, item.fromSurfaceId)}
            </td>
            <td className="px-2 py-1">
              {regionSurface(item.toRegionId, item.toSurfaceId)}
            </td>
          </tr>)}</tbody>
        </table>
      </div>
    </details>
  </div>;
}
