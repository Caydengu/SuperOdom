#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "$script_dir/.." && pwd)"
sdk_root="${NATNET_SDK_ROOT:-}"
output_path="${1:-$repository_root/build/optitrack_reference_recorder}"

if [[ -z "$sdk_root" ]]; then
  echo "Set NATNET_SDK_ROOT to the extracted NatNet_SDK_4.4 Ubuntu directory." >&2
  exit 2
fi

header="$sdk_root/include/NatNetClient.h"
library="$sdk_root/lib/libNatNet.so"
source_file="$repository_root/tools/optitrack_reference_recorder.cpp"

for required_file in "$header" "$library" "$source_file"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required file: $required_file" >&2
    exit 3
  fi
done

mkdir -p "$(dirname "$output_path")"
g++ \
  -std=c++17 \
  -O2 \
  -Wall \
  -Wextra \
  -Wpedantic \
  -I"$sdk_root/include" \
  "$source_file" \
  -L"$sdk_root/lib" \
  -lNatNet \
  -pthread \
  -o "$output_path"

echo "$output_path"
