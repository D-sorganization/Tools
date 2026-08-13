/** Glossary entry type — shared by the split data modules. */

export interface GlossaryEntry {
  /** Title Case display name. */
  term: string;
  /** 1-3 sentence sourced definition. */
  definition: string;
}
