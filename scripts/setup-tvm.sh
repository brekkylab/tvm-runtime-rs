#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 1. Parameter Handling
GIT_REPOSITORY="${1:-https://github.com/brekkylab/relax.git}"
GIT_TAG="${2:-brekky}"
TARGET_PATH="$SCRIPT_DIR/../3rdparty/tvm"

echo "========================================"
echo "▶ Repository : $GIT_REPOSITORY"
echo "▶ Tag/Branch : $GIT_TAG"
echo "========================================"

cd "$TARGET_PATH" || { echo "❌ Failed to enter directory"; exit 1; }
echo "🔖 Fetching and checking out tag: $GIT_TAG"
git fetch --all --tags
git checkout "$GIT_TAG"
echo "🔗 Updating nested submodules..."
git submodule update --init --recursive

cd - > /dev/null

echo "🚚 Copying Rust FFI components to project root..."
FFI_SOURCE_BASE="$TARGET_PATH/3rdparty/tvm-ffi/rust"
COMPONENTS=("tvm-ffi" "tvm-ffi-sys" "tvm-ffi-macros")

for COMPONENT in "${COMPONENTS[@]}"; do
    SRC="$FFI_SOURCE_BASE/$COMPONENT"
    if [ -d "$SRC" ]; then
        cp -r "$SRC" .
        echo "  - ✅ Copied $COMPONENT"
    else
        echo "  - ⚠️  Warning: Source $SRC not found!"
    fi
done

echo "🧹 Cleaning up build scripts..."
[ -f "tvm-ffi-sys/build.rs" ] && rm "tvm-ffi-sys/build.rs" && echo "  - 🗑️ Removed tvm-ffi-sys/build.rs"
[ -f "tvm-ffi/build.rs" ] && rm "tvm-ffi/build.rs" && echo "  - 🗑️ Removed tvm-ffi/build.rs"

echo "✨ Task completed successfully!"
