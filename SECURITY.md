# Security policy

UFF is a local scientific analysis package. It does not require credentials,
network services, or execution of code embedded in data files.

## Supported version

Security fixes target the latest `4.x` release line.

## Reporting

Report a vulnerability privately through GitHub's repository security advisory
interface. Do not include sensitive data in a public issue.

## Data-handling notes

- CSV inputs are parsed as data and are never evaluated as Python.
- Output paths are supplied by the local user.
- JSON receipts may contain input paths and repeated scalar metadata; review
  them before public release if local paths are sensitive.
- UFF does not download SPARC or other datasets automatically.
