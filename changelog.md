# Changelog

All notable changes to algorithms, weights, and underlying OSO models will be documented here.

## [M3] - 2025-05-19

### Added
- Generated utility labels for all devtooling projects.
- Added `utility_weights` parameter to `devtooling__arcturus.yaml` to control utility label weights.
- Included World Verified Users in the `qualified_addresses_monthly` metric within `onchain__goldilocks.yaml`.

### Changed
- Reduced `link_type_weights` for `package_dependency` from 3.0 to 1.5 in `devtooling__arcturus.yaml` to reflect high utility weightings for packages.
- Increased the `percentile_cap` in `onchain__goldilocks.yaml` from 97 to 98.

### Fixed
- Corrected OSO-side logic for building the developer graph: developers tied to onchain contracts in separate GitHub organizations no longer receive links to devtooling projects.
- Improved handling of projects that share a root deployer but deploy contracts across multiple repositories.
- Fixed one case of a DefiLlama slug being misattributed to the wrong project.

## [M2] - 2025-04-28

### Added
- Introduced new onchain-builder weighting metrics options (e.g., World Verified Users and Account Abstraction UserOps); these currently have no explicit weightings in M2.
- Added Worldchain-specific event handling for UserOps.
- Developed utility scripts for fetching OSO data and generating algorithm results.
- Configured serialization of each round’s results to JSON under `data/outputs`.

### Changed
- Moved budget allocation settings into the `allocation` section of each algorithm config file.
- Updated amortized contract invocations to include all relevant account abstraction events.
- Simplified amortization logic so that projects receive equal credit per invocation, regardless of how many other Retro Funded projects are invoked in the same transaction.

### Fixed
- Manually linked several GitHub repositories to their corresponding NPM devtooling packages.
- Ensured distinct collections of projects per measurement period based on application submission dates.