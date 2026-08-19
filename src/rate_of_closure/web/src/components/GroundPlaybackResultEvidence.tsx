/** Accessible raw trajectory and event ledgers for one immutable ground result. */

import type {
  FlightToGroundResult,
  GroundVec3,
} from "../model/flightGroundTypes";

type EvidenceSubject = "primary" | "comparison";

function VectorCells({ vectors }: { readonly vectors: readonly GroundVec3[] }) {
  return (
    <>
      {vectors.flatMap((vector, group) =>
        vector.map((value, axis) => (
          <td key={`${group}-${axis}`}>{value.toFixed(6)}</td>
        )),
      )}
    </>
  );
}

export function GroundPlaybackResultEvidence({
  result,
  subject,
}: {
  readonly result: FlightToGroundResult;
  readonly subject: EvidenceSubject;
}) {
  const start = result.trajectory[0].time_s;
  const titleSubject = subject[0].toUpperCase() + subject.slice(1);
  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <section className="overflow-x-auto rounded-lg border border-slate-800 p-3">
        <h3 className="mb-2 font-semibold">
          {titleSubject} trajectory samples
        </h3>
        <table
          className="min-w-full text-left text-xs"
          aria-label={`Ground ${subject} trajectory evidence`}
        >
          <thead>
            <tr>
              {[
                "Sample",
                "Absolute s",
                "Elapsed s",
                "Phase",
                "x m",
                "y m",
                "z m",
                "vx m/s",
                "vy m/s",
                "vz m/s",
                "ωx rad/s",
                "ωy rad/s",
                "ωz rad/s",
              ].map((label) => (
                <th scope="col" key={label} className="pr-3">
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {result.trajectory.map((point, index) => (
              <tr key={`${point.time_s}-${index}`}>
                <th scope="row">{index}</th>
                <td>{point.time_s.toFixed(6)}</td>
                <td>{(point.time_s - start).toFixed(6)}</td>
                <td>{point.phase}</td>
                <VectorCells
                  vectors={[
                    point.position_m,
                    point.velocity_m_s,
                    point.angular_velocity_rad_s,
                  ]}
                />
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <section className="overflow-x-auto rounded-lg border border-slate-800 p-3">
        <h3 className="mb-2 font-semibold">{titleSubject} event ledger</h3>
        <table
          className="min-w-full text-left text-xs"
          aria-label={`Ground ${subject} event evidence`}
        >
          <thead>
            <tr>
              {[
                "Sequence",
                "Event",
                "Time s",
                "x m",
                "y m",
                "z m",
                "vx before m/s",
                "vy before m/s",
                "vz before m/s",
                "vx after m/s",
                "vy after m/s",
                "vz after m/s",
                "ωx before rad/s",
                "ωy before rad/s",
                "ωz before rad/s",
                "ωx after rad/s",
                "ωy after rad/s",
                "ωz after rad/s",
              ].map((label) => (
                <th scope="col" key={label} className="pr-3">
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {result.events.map((event) => (
              <tr key={event.sequence}>
                <th scope="row">{event.sequence}</th>
                <td>{event.event_type}</td>
                <td>{event.time_s.toFixed(6)}</td>
                <VectorCells
                  vectors={[
                    event.position_m,
                    event.velocity_before_m_s,
                    event.velocity_after_m_s,
                    event.angular_velocity_before_rad_s,
                    event.angular_velocity_after_rad_s,
                  ]}
                />
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
