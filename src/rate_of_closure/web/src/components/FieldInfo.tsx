import { useId } from "react";

export function FieldInfo({ label, guidance }: { label: string; guidance: string }) {
  const descriptionId = useId();
  return (
    <details className="group relative inline-block">
      <summary
        className="ml-1 inline-flex cursor-pointer list-none items-center gap-1 rounded-md border border-sky-500/35 bg-sky-500/10 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-sky-300 transition hover:border-sky-400 hover:bg-sky-500/20 focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400"
        title={`Explain ${label}`}
        aria-label={`Explain ${label}`}
        aria-describedby={descriptionId}
      >
        <span aria-hidden="true">ⓘ</span>
        Details
      </summary>
      <p id={descriptionId} className="absolute left-0 top-full z-30 mt-1 w-72 max-w-[80vw] rounded-lg border border-sky-500/35 bg-slate-950 p-3 text-left text-xs font-normal normal-case tracking-normal text-slate-300 shadow-2xl shadow-black/50">
        {guidance}
      </p>
    </details>
  );
}
