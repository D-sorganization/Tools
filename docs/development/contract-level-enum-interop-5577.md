# Contract-Level Enum Interoperability (#5577)

## Design

The shared contracts module owns the canonical ContractLevel enum and stores
only its members in runtime state and the backward-compatible module aliases.
set_contract_level accepts either that enum or another enum.Enum whose value
exactly matches one of the canonical values (enforce, warn, or off). It converts
the value to the local enum before changing state or aliases. This allows
duplicate enum definitions loaded by compatible package entry points to
interoperate without storing a foreign enum and silently bypassing
identity-based enforcement branches.

A non-enum argument raises TypeError; an enum with an unsupported value raises
ValueError. Both failures leave runtime state and aliases unchanged. The
environment-based default resolver, contract behavior at each enforcement
level, and set_contracts_enabled mapping are unchanged.

## Turnover

The regression covers a separate foreign enum at each supported value. It
checks canonical local enum identity and the existing ENFORCE/WARN/OFF behavior,
then verifies that non-enum and unsupported enum inputs do not mutate state or
aliases.

Validation on Python 3.13.3: the five new focused cases passed, and the shared
contracts plus cross-repository import compatibility suites passed 114 tests
with eight existing deprecation warnings. Ruff check and format were applied
to the two changed Python files. No calculation, physics pathway, or default
policy change is included.

## Full pre-PR gate

The first full-gate launcher invocation did not complete successfully. The
affected suite had 415 passed and one collection error; consistency checks had
three collection errors, while other gates ran. The collection errors came from
adding the Repository_Management checkout to the child process `PYTHONPATH`,
which shadowed the Tools namespace package. The launcher was corrected to use
parent-import-only setup, then the full gate completed with actual exit code 0:
425 affected tests passed and 10 existing warnings were reported. This validates
the unchanged tested source at commit
`462c56eba224cb784562d5a8ced271c24e373cf5`.

The full-gate receipt is stored in the OGLaptop LOCAL planning workspace at
`C:/Users/diete/Repositories/docs/development/feedback_controls_planning/tools5577_fullprepr_20261012/prepr.parent.json`
(SHA-256 `3b2dd339070b9f74e7f46c0947b5d139cb8ba2f6e097b39b709086cf9d19de0d`);
stdout SHA-256 is `1617ab5099fd62072ddab8f95cf7e2fe4572c6163cb3506c7155af2f06dc4e3c`.

Git commit and push used normal commands. Hooks were unavailable: the primary
Tools `.git/config` sets `core.hooksPath` to `C:\Users\diete\Repositories\Tools\nul`,
which does not exist as a hooks directory. No hook pass is claimed.

The scope remains canonical enum interoperability only. The default optimized
policy is preserved, and no product code or broader epic implementation is
included in this turnover.
