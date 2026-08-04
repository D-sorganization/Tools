/**
 * Calculation Description tab — the calculation, step by step.
 *
 * Renders model/derivation.ts with KaTeX (bundled locally; no CDN),
 * substituting the live scenario so every number in the results panel
 * can be traced to the rigid-body kinematics that produced it.
 */

import katex from "katex";
import "katex/dist/katex.min.css";
import { useMemo } from "react";

import { derivationSteps } from "../model/derivation";
import type { ImpactScenario } from "../model/impact";

function Formula({ tex }: { tex: string }) {
  const html = useMemo(
    () =>
      katex.renderToString(tex, {
        throwOnError: false,
        displayMode: true,
      }),
    [tex],
  );
  return (
    <div
      className="overflow-x-auto py-1 text-slate-100"
      // KaTeX output is generated locally from our own strings.
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

export function Derivation({ scenario }: { scenario: ImpactScenario }) {
  const steps = useMemo(() => derivationSteps(scenario), [scenario]);
  return (
    <section
      aria-label="Derivation and traceability"
      className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-6 shadow-lg shadow-black/20 backdrop-blur"
    >
      <p className="mb-4 max-w-3xl text-sm text-slate-400">
        Every result traces to one of the steps below; the second line of
        each formula substitutes the current scenario. Sources: the
        AffineDrift Launch Monitor Technology Review (frame and sign
        conventions), the closure-rate derivation (d / R_ISA, deg/ft), and
        the Cheetham 2014 closure-rate dossier.
      </p>
      <ol className="space-y-6">
        {steps.map((step, index) => (
          <li key={step.title}>
            <h3 className="text-sm font-semibold tracking-wide text-sky-200">
              Step {index + 1} — {step.title}
            </h3>
            <p className="mt-1 max-w-3xl text-sm text-slate-400">
              {step.narrative}
            </p>
            <Formula tex={step.latex} />
            <Formula tex={step.values} />
            {index < steps.length - 1 && (
              <hr className="mt-4 border-slate-800" />
            )}
          </li>
        ))}
      </ol>
    </section>
  );
}
