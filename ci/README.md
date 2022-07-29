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


### Release builds

AZP release builds

### Service Connection

In order to create issues after nightly build failure, the AZP nightly pipeline uses a service connection
with authorization to post to GitHub.
  - Configuration page: [https://dev.azure.com/TileDB-Inc/CI/_settings/adminservices](https://dev.azure.com/TileDB-Inc/CI/_settings/adminservices)
  - Connection name: `TileDB-Py-Test`
  - Setup, from services configuration page above:
    - Create a new Personal Access token
      - At present, the repo dispatch token must be linked to an individual user account.
      - Visit: https://github.com/settings/tokens/new
      - Create token with `public_repo` scope and 1 year expiration (maximum)
    - Visit configuration page linked above
      - Select `New Service Connection`
      - Select `Generic`
        - Server URL: `https://api.github.com/`
        - Password/token: <GitHub personal access token created above>
      - Disable access to all repositories (will require pipeline-specific authorization on first pipeline execution)
    - Save the new connection. Note that the connection name must match the name specified for
      `connectedServieName` in `misc/azure-release.yml`. Note that you
