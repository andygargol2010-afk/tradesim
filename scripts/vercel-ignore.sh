#!/usr/bin/env bash
# Vercel "Ignored Build Step" (vercel.json -> ignoreCommand). Exit 0 = skip the build, 1 = build.
#
# The scheduled runs commit only live data (live/, dashboard/live.json) and the dashboard reads that
# data straight from the repository, so those commits need no deployment. Build whenever anything
# else changed since the last successful deployment of this branch (VERCEL_GIT_PREVIOUS_SHA).
# When in doubt (no previous deployment, shallow clone without it) build.
base="${VERCEL_GIT_PREVIOUS_SHA:-}"
if [ -z "$base" ] || ! git cat-file -e "${base}^{commit}" 2>/dev/null; then
  echo "no previous deployment to compare with: build"
  exit 1
fi
if git diff --quiet "$base" HEAD -- . ':(exclude)live' ':(exclude)dashboard/live.json'; then
  echo "only live data changed since ${base:0:7}: skip build (the dashboard reads it from the repository)"
  exit 0
fi
echo "code changed since ${base:0:7}: build"
exit 1
