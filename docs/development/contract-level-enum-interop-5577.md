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