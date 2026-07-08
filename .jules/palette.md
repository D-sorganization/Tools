## 2024-07-08 - Native Form Submission for Data Entry
**Learning:** React data-entry forms relying on custom `onClick` handlers for submit buttons break native "Enter" key submission, which is a major accessibility anti-pattern.
**Action:** When creating data-entry calculators or settings panels, wrap inputs and the submit button in a native `<form onSubmit={...}>` element instead of binding `onClick` directly to the button. This enables native "Enter" key submission for keyboard accessibility.
