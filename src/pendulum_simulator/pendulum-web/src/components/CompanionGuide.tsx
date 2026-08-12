import { useMemo, useState } from "react";

import {
  COMPANION_CATALOG,
  type CompanionModel,
  searchGlossary,
} from "../companionCatalog";
import "./CompanionGuide.css";

interface CompanionGuideProps {
  onSelectModel?: (model: CompanionModel) => void;
}

function BulletList({ items }: { items: string[] }) {
  return (
    <ul>
      {items.map((item) => (
        <li key={item}>{item}</li>
      ))}
    </ul>
  );
}

export function CompanionGuide({ onSelectModel }: CompanionGuideProps) {
  const [isOpen, setIsOpen] = useState(true);
  const [experimentId, setExperimentId] = useState(
    COMPANION_CATALOG.experiments[0].id,
  );
  const [glossaryQuery, setGlossaryQuery] = useState("");
  const experiment = COMPANION_CATALOG.experiments.find(
    (item) => item.id === experimentId,
  );
  const terms = useMemo(() => searchGlossary(glossaryQuery), [glossaryQuery]);

  if (!experiment) {
    throw new Error(`unknown companion experiment: ${experimentId}`);
  }

  return (
    <section className="companion-guide" aria-labelledby="companion-guide-title">
      <div className="companion-guide__heading">
        <div>
          <h2 id="companion-guide-title">Proximal–Distal Companion Guide</h2>
          <p>
            Explore a declared hypothesis, its observables, and what would challenge
            the interpretation before changing model inputs.
          </p>
        </div>
        <button
          className="btn btn-secondary"
          type="button"
          aria-expanded={isOpen}
          aria-controls="companion-guide-content"
          onClick={() => setIsOpen((current) => !current)}
        >
          {isOpen ? "Hide Guide" : "Show Guide"}
        </button>
      </div>
      {isOpen && (
        <div id="companion-guide-content" className="companion-guide__content">
          <div className="companion-guide__experiment">
            <label htmlFor="companion-experiment">Guided Experiment</label>
            <select
              id="companion-experiment"
              value={experimentId}
              onChange={(event) => setExperimentId(event.target.value)}
            >
              {COMPANION_CATALOG.experiments.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.title}
                </option>
              ))}
            </select>
            <button
              className="btn btn-primary"
              type="button"
              onClick={() => onSelectModel?.(experiment.model)}
              title={`Switch To The ${experiment.model} Model`}
            >
              Open {experiment.model === "golfer" ? "Golfer" : experiment.model} Model
            </button>
            <h3>Purpose</h3>
            <p>{experiment.purpose}</p>
            <h3>Hypothesis</h3>
            <p>{experiment.hypothesis}</p>
            <h3>What Would Challenge This Result?</h3>
            <p>{experiment.falsifier}</p>
          </div>
          <div className="companion-guide__instructions">
            <h3>Workflow</h3>
            <BulletList items={experiment.workflow} />
            <h3>Tips</h3>
            <BulletList items={experiment.tips} />
            <h3>Limitations</h3>
            <BulletList items={experiment.limitations} />
          </div>
          <div className="companion-guide__glossary">
            <label htmlFor="companion-glossary-search">Glossary</label>
            <input
              id="companion-glossary-search"
              aria-label="Search The Glossary"
              type="search"
              value={glossaryQuery}
              placeholder="Search Terms And Definitions"
              onChange={(event) => setGlossaryQuery(event.target.value)}
            />
            <dl>
              {terms.map((term) => (
                <div key={term.id} className="companion-guide__term">
                  <dt title={term.caution}>{term.term}</dt>
                  <dd>
                    {term.plain_language} <span>({term.units})</span>
                  </dd>
                </div>
              ))}
            </dl>
          </div>
        </div>
      )}
    </section>
  );
}
