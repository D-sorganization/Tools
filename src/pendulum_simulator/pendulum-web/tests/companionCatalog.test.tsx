import { renderToString } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { CompanionGuide } from "../src/components/CompanionGuide";
import {
  COMPANION_CATALOG,
  searchGlossary,
} from "../src/companionCatalog";

describe("proximal-distal companion catalog", () => {
  it("covers every interactive model with falsifiable experiments", () => {
    const models = new Set(COMPANION_CATALOG.experiments.map((item) => item.model));

    expect(models).toEqual(new Set(["double", "triple", "golfer"]));
    expect(COMPANION_CATALOG.experiments.length).toBeGreaterThanOrEqual(6);
    for (const experiment of COMPANION_CATALOG.experiments) {
      expect(experiment.falsifier.length).toBeGreaterThan(10);
      expect(experiment.workflow.length).toBeGreaterThan(0);
      expect(experiment.tips.length).toBeGreaterThan(0);
    }
  });

  it("searches definitions as well as glossary labels", () => {
    const ids = searchGlossary("counterfactual").map((term) => term.id);

    expect(ids).toContain("ztcf");
    expect(ids).toContain("forward-counterfactual");
  });

  it("renders an accessible guide with workflow, tips, and limitations", () => {
    const html = renderToString(<CompanionGuide />);

    expect(html).toContain("Proximal–Distal Companion Guide");
    expect(html).toContain("Workflow");
    expect(html).toContain("Tips");
    expect(html).toContain("What Would Challenge This Result?");
    expect(html).toContain("Limitations");
    expect(html).toContain('aria-label="Search The Glossary"');
  });
});
