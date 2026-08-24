/**
 * Calculation Description tab — every model's derivation, step by step.
 *
 * Renders model/derivationModels.ts with KaTeX (bundled locally; no
 * CDN): the closure chain, the impact model, the active ball-flight
 * model, and — when a pendulum swing source is configured — the swing
 * model, substituting the live scenario so every number in the results
 * panel can be traced to the physics that produced it.
 */

import DOMPurify from "dompurify";
import katex from "katex";
import "katex/dist/katex.min.css";
import { useMemo } from "react";

import {
  DEFAULT_DERIVATION_CONFIG,
  derivationSections,
  type DerivationConfig,
} from "../model/derivationModels";
import type { ImpactScenario } from "../model/impact";

function Formula({ tex }: { tex: string }) {
  const html = useMemo(
    () =>
      // Defense in depth: Sanitize KaTeX output to prevent XSS vulnerabilities
      // before injecting it directly into the DOM via dangerouslySetInnerHTML.
      DOMPurify.sanitize(
        katex.renderToString(tex, {
          throwOnError: false,
          displayMode: true,
        }),
      ),
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

export function Derivation({
  scenario,
  config = DEFAULT_DERIVATION_CONFIG,
}: {
  scenario: ImpactScenario;
  config?: DerivationConfig;
}) {
  const sections = useMemo(
    () => derivationSections(scenario, config),
    [scenario, config],
  );
  return (
    <section
      aria-label="Calculation description"
      className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-6 shadow-lg shadow-black/20 backdrop-blur"
    >
      <p className="mb-4 max-w-3xl text-sm text-slate-400">
        Every number in the app traces to one of the sections below — the
        closure chain, the impact model, the active ball-flight model,
        and (when a pendulum source is configured) the swing model. The
        second line of each formula substitutes the current scenario.
        Sources: the AffineDrift Launch Monitor Technology Review, the
        Cheetham 2014 closure-rate dossier, and the swing_sim
        impact/flight/reference derivations.
      </p>
      {sections.map((section) => (
        <section key={section.key} aria-label={section.title} className="mb-8">
          <h2 className="text-base font-bold tracking-wide text-sky-300">
            {section.title}
          </h2>
          <p className="mb-3 mt-1 max-w-3xl text-sm text-slate-400">
            {section.intro}
          </p>
          <ol className="space-y-6">
            {section.steps.map((step, index) => (
              <li key={step.title}>
                <h3 className="text-sm font-semibold tracking-wide text-sky-200">
                  Step {index + 1} — {step.title}
                </h3>
                <p className="mt-1 max-w-3xl text-sm text-slate-400">
                  {step.narrative}
                </p>
                <Formula tex={step.latex} />
                <Formula tex={step.values} />
                {index < section.steps.length - 1 && (
                  <hr className="mt-4 border-slate-800" />
                )}
              </li>
            ))}
          </ol>
        </section>
      ))}
    </section>
  );
}
