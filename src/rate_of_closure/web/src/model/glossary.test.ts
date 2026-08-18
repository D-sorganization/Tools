/**
 * Glossary parity + behaviour tests (#4120 V4).
 *
 * The fixture (glossary.fixture.json) is generated from the Python
 * glossary and re-checked by tests/rate_of_closure/test_glossary_parity
 * on the pytest side, so the two glossaries cannot drift apart.
 */

import { describe, expect, it } from "vitest";

import fixture from "./glossary.fixture.json";
import { FIELD_TO_TERM, GLOSSARY, searchTerms } from "./glossary";

describe("glossary parity with Python", () => {
  it("pins the term keys against the Python-generated fixture", () => {
    expect(Object.keys(GLOSSARY)).toEqual(fixture.keys);
  });

  it("pins the field-to-term mapping", () => {
    expect(FIELD_TO_TERM).toEqual(fixture.field_terms);
  });

  it("every mapped term exists", () => {
    for (const term of Object.values(FIELD_TO_TERM)) {
      expect(GLOSSARY[term]).toBeDefined();
    }
  });
});

describe("glossary content", () => {
  it("definitions are substantive and sourced", () => {
    for (const [key, entry] of Object.entries(GLOSSARY)) {
      expect(entry.definition.length, key).toBeGreaterThanOrEqual(60);
      expect(entry.term.trim(), key).not.toBe("");
      expect(entry.definition, key).toContain("(");
    }
  });

  it("search filters case-insensitively and empty returns all", () => {
    expect(searchTerms("")).toEqual(Object.keys(GLOSSARY));
    const hits = searchTerms("cheetham");
    expect(hits.length).toBeGreaterThan(0);
    expect(hits.length).toBeLessThan(Object.keys(GLOSSARY).length);
    expect(hits).toContain("ccv");
    expect(searchTerms("zzz-no-such-term")).toEqual([]);
  });
});
