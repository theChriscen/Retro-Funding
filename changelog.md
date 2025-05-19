# Change Log

All notable changes to algorithms, weights, and underlying OSO models will be documented here.

## [M3] - 2025-05-19

### Added

- Generated utility labels for all devtooling projects.
- `utility_weights` parameter to `devtooling__arcturus.yaml` to control the weights of utility labels.
- World Verified Users to `qualified_addresses_monthly` metric in `onchain__goldilocks.yaml`.

### Changed

- `link_type_weights` for `package_dependency` have been reduced from 3.0 to 1.5 in `devtooling__arcturus.yaml` as packages also receive high utility weightings.
- `percentile_cap` in `onchain__goldilocks.yaml` has been increased from 97 to 98.

### Fixed

- OSO-side logic for building the developer graph for projects that have onchain activity tied to contracts in a sepearate GitHub organization. Now, developers that are associated with the onchain project do not receive a link to the devtooling project.
- OSO-side handling for projects that share a root deployer but have contracts spread over more than one project.
- One case where a DefiLlama slug was misattributed to a project.

## [M2] - 2025-04-28

### Added

- New onchain builder weighting metrics options, eg, World Verified Users and Account Abstraction UserOps. Note: these are not yet given any explicit weighting in the M2 algorithms.
- World Chain specific event handling (as UserOps).
- Various utility scripts for fetching data from OSO and generating results.
- Results for each round are serialized to JSON and saved in the appropriate `data/outputs` directory.

### Changed

- Budget allocation settings are now handled in the `allocation` section of each algorithm config file.
- Amortized contract invocations include all relevant account abstraction events
- Amortization logic for contract interactions and gas fees has been simplified; now a project will be credited with the same amount of impact regardless of how many other Retro Funded projects are invoked in the same transaction.

### Fixed

- Manually link several GitHub repos to their corresponding devtooling packages on NPM.
- OSO-side handling for creating distinct collections of projects for each measurement period (based on application submission date).