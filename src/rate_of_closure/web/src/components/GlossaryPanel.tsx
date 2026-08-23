/**
 * Glossary section (#4120 V4) — searchable term list + definition pane.
 *
 * Mirrors the PyQt6 Glossary tab over model/glossary.ts: a search box
 * filters the term list live, selecting a term shows its sourced
 * definition, and explanation cards deep-link here with a pre-selected
 * term.
 */

import { useMemo, useState } from "react";

import { GLOSSARY, searchTerms } from "../model/glossary";

export function GlossaryPanel({ initialTerm }: { initialTerm?: string }) {
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<string>(
    initialTerm && initialTerm in GLOSSARY
      ? initialTerm
      : Object.keys(GLOSSARY)[0],
  );
  const keys = useMemo(() => searchTerms(query), [query]);
  const entry = GLOSSARY[selected];

  return (
    <section
      aria-label="Glossary"
      className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-6 shadow-lg shadow-black/20 backdrop-blur"
    >
      <p className="mb-4 max-w-3xl text-sm text-slate-400">
        Every technical term used in the app, with sourced definitions.
        Type to filter; click a term to read its definition. Explanation
        cards across the app link here.
      </p>
      <div className="grid gap-6 lg:grid-cols-[300px_1fr]">
        <div className="min-w-0">
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search terms and definitions…"
            title="Filter the glossary: matches term names and definition text, case-insensitive"
            aria-label="Search glossary"
            className="mb-3 w-full rounded border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-slate-100 focus:border-blue-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
          />
          <ul
            aria-label="Glossary terms"
            className="max-h-[28rem] space-y-1 overflow-y-auto pr-1"
          >
            {keys.map((key) => (
              <li key={key}>
                <button
                  type="button"
                  onClick={() => setSelected(key)}
                  aria-pressed={selected === key}
                  title={GLOSSARY[key].definition}
                  className={
                    "w-full truncate rounded-md border px-3 py-1.5 text-left text-sm transition-all " +
                    (selected === key
                      ? "border-sky-400/60 bg-sky-500/15 text-sky-200"
                      : "border-slate-800/80 bg-slate-900/50 text-slate-300 hover:border-slate-600")
                  }
                >
                  {GLOSSARY[key].term}
                </button>
              </li>
            ))}
            {keys.length === 0 && (
              <li className="px-3 py-1.5 text-sm text-slate-500">
                No matching term — clear the search to see the full
                glossary.
              </li>
            )}
          </ul>
        </div>
        <div className="min-w-0">
          {entry && (
            <>
              <h3 className="text-lg font-semibold text-sky-200">
                {entry.term}
              </h3>
              <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-300">
                {entry.definition}
              </p>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
