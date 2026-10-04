# Chile regulation context (dated research note)

**Status date:** 2026-10-04. This note is documentation context only. It is not a hard-coded legal rule engine, legal advice, actuarial certification, or a guarantee that a calculation reflects current law.

## Official sources consulted

- BCN Ley Chile, Decreto Ley N° 3.500, *Establece Nuevo Sistema de Pensiones*: <https://www.bcn.cl/leychile/navegar?idNorma=7147>
- BCN Ley Chile, Ley N° 21.735, 2025 pension reform: <https://www.bcn.cl/leychile/navegar?idNorma=1212060>
- Superintendencia de Pensiones, Ley N° 21.735 normative page: <https://www.spensiones.cl/portal/institucional/594/w3-article-16483.html>

## Why this matters for `afp-operations`

- DL 3.500 is the base Chilean individual-capitalization pension-system statute.
- Ley 21.735, published 2025-03-26, creates a mixed pension system and social insurance in the contributory pillar, improves PGU, and modifies regulatory structures.
- BCN notes that Ley 21.735 has a general 2027-04-01 effectiveness frame with transitional exceptions.

## Package rule

The package contains explicit dated research fixtures. It must not present those fixtures as official Chilean law. A future legally validated rules layer would need its own source registry, legal review, versioning, and tests.

## Implementation posture

- `afp-operations demo` uses synthetic values.
- Contribution splits and quotes are software-contract examples unless a separately validated policy provider supplies official parameters.
- No live AFP operations, custody, banking, or official benefit determinations are implemented.
