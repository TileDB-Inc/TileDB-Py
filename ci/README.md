# CI documentation

TileDB-Py currently uses two CI services:
  - GitHub actions for routine CI
  - Azure Pipelines for release packaging

## GitHub Actions

<TODO>

### Nightly Builds

- <triggering setup>
## Azure Pipelines

AZP is configured in [azure-pipelines.yml](), which points to two pipeline files in `misc`:
  - [misc/azure-ci.yml](): Legacy CI, to ensure that the release build continue to work.
  - [misc/azure-release.yml](): release build system, which creates Python wheels for PyPI distribution.



### Service Connection

In order to create issues after nightly build failure, the AZP nightly pipeline uses a service connection
with authorization to post to GitHub.
  - Configuration page: [https://dev.azure.com/TileDB-Inc/CI/_settings/adminservices](https://dev.azure.com/TileDB-Inc/CI/_settings/adminservices)
  - Connection name: `TileDB-Py-Test`
