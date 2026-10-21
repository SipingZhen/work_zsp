#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
cmake -S . -B build
cmake --build build
if ldd noetix_joint_controller | grep -q 'not found'; then
  echo 'Controller runtime libraries are missing:' >&2
  ldd noetix_joint_controller | grep 'not found' >&2
  exit 1
fi
chmod +x noetix_joint_controller

