---
title: Nightly GitHub Actions Build Fail
assignees: nguyenv, ihnorton
labels: bug
---

The nightly GitHub Actions build failed on {{ date | date('ddd, MMMM Do YYYY') }}.
See run for more details:
https://github.com/{{ env.GITHUB_REPOSITORY }}/actions/runs/{{ env.GITHUB_RUN_ID }}
