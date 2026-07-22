import React from "react";

/**
 * A tiny, dependency-free Markdown renderer for the in-app help.
 *
 * The CSP on the HMI blocks external libraries, and the help content only needs
 * a small, well-defined subset of Markdown, so this renders that subset directly
 * to React elements (DRY: one renderer for every tab's help). Supported:
 *   - `#`..`####` headings
 *   - `-` / `*` unordered lists and `1.` ordered lists
 *   - `---` horizontal rule
 *   - blank-line-separated paragraphs
 *   - inline `**bold**` and `` `code` ``
 *
 * Anything else is treated as paragraph text, so unknown syntax degrades to
 * plain prose rather than throwing.
 */

const INLINE = /(\*\*([^*]+)\*\*|`([^`]+)`)/g;

/** Render inline `**bold**` and `` `code` `` spans within one line of text. */
export function renderInline(text: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  let last = 0;
  let key = 0;
  let m: RegExpExecArray | null;
  INLINE.lastIndex = 0;
  while ((m = INLINE.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    if (m[2] !== undefined) {
      nodes.push(<strong key={key++}>{m[2]}</strong>);
    } else if (m[3] !== undefined) {
      nodes.push(<code key={key++}>{m[3]}</code>);
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

const HEADING_TAGS = ["h2", "h3", "h4", "h5"] as const;

/**
 * Render a Markdown string to React nodes.
 *
 * @throws TypeError if `md` is not a string (DbC — a caller bug, not content).
 */
export function renderMarkdown(md: string): React.ReactElement {
  if (typeof md !== "string") {
    throw new TypeError("renderMarkdown: md must be a string");
  }
  const lines = md.replace(/\r\n/g, "\n").split("\n");
  const blocks: React.ReactNode[] = [];
  let key = 0;
  let para: string[] = [];
  let i = 0;

  const flushPara = (): void => {
    if (para.length > 0) {
      blocks.push(<p key={key++}>{renderInline(para.join(" "))}</p>);
      para = [];
    }
  };

  while (i < lines.length) {
    const t = lines[i].trim();

    if (t === "") {
      flushPara();
      i++;
      continue;
    }
    if (t === "---") {
      flushPara();
      blocks.push(<hr key={key++} />);
      i++;
      continue;
    }

    const heading = /^(#{1,4})\s+(.*)$/.exec(t);
    if (heading) {
      flushPara();
      const Tag = HEADING_TAGS[heading[1].length - 1];
      blocks.push(<Tag key={key++}>{renderInline(heading[2])}</Tag>);
      i++;
      continue;
    }

    if (/^[-*]\s+/.test(t)) {
      flushPara();
      const items: React.ReactNode[] = [];
      while (i < lines.length && /^[-*]\s+/.test(lines[i].trim())) {
        const item = lines[i].trim().replace(/^[-*]\s+/, "");
        items.push(<li key={items.length}>{renderInline(item)}</li>);
        i++;
      }
      blocks.push(<ul key={key++}>{items}</ul>);
      continue;
    }

    if (/^\d+\.\s+/.test(t)) {
      flushPara();
      const items: React.ReactNode[] = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim())) {
        const item = lines[i].trim().replace(/^\d+\.\s+/, "");
        items.push(<li key={items.length}>{renderInline(item)}</li>);
        i++;
      }
      blocks.push(<ol key={key++}>{items}</ol>);
      continue;
    }

    para.push(t);
    i++;
  }
  flushPara();
  return <>{blocks}</>;
}
