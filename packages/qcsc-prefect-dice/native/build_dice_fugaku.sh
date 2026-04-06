#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DICE_REPO_URL="${DICE_REPO_URL:-https://github.com/caleb-johnson/Dice.git}"
DICE_REF="${DICE_REF:-}"
BOOST_URL="${BOOST_URL:-https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.gz}"
BOOST_DIR_NAME="${BOOST_DIR_NAME:-boost_1_85_0}"
FUGAKU_MPI_SPEC="${FUGAKU_MPI_SPEC:-fujitsu-mpi@head-gcc8}"
DICE_BIN_PATH="${DICE_BIN_PATH:-$SCRIPT_DIR/bin}"
NPROC="${NPROC:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"

. /vol0004/apps/oss/spack/share/spack/setup-env.sh
spack load "$FUGAKU_MPI_SPEC"

rm -rf external/
mkdir -p external

wget "$BOOST_URL" -O external/"${BOOST_DIR_NAME}.tar.gz"
tar -xzf external/"${BOOST_DIR_NAME}.tar.gz" -C "$SCRIPT_DIR"/external
rm -f external/"${BOOST_DIR_NAME}.tar.gz"

git clone "$DICE_REPO_URL" "$SCRIPT_DIR"/external/Dice
if [ -n "$DICE_REF" ]; then
    git -C "$SCRIPT_DIR"/external/Dice checkout "$DICE_REF"
fi

export BOOST_ROOT="$SCRIPT_DIR"/external/"$BOOST_DIR_NAME"
DICE_ROOT="$SCRIPT_DIR"/external/Dice

cd "$BOOST_ROOT"
echo "using mpi ;" > user-config.jam
./bootstrap.sh --with-libraries=serialization,mpi
./b2 -j"$NPROC" --user-config=user-config.jam --prefix="$BOOST_ROOT"/stage

cd "$DICE_ROOT"
sed -i.bak 's/HAS_AVX2 = yes/HAS_AVX2 = no/' Makefile || true
sed -i.bak 's/#-I$(BOOST)/-I$(BOOST)/' Makefile || true
make -j"$NPROC" Dice

mkdir -p "$DICE_BIN_PATH"
cp "$DICE_ROOT"/bin/Dice "$DICE_BIN_PATH"
cp "$BOOST_ROOT"/stage/lib/*.so* "$DICE_BIN_PATH"

echo "Build completed: $DICE_BIN_PATH/Dice"
echo "If you run Dice outside qcsc-prefect, use:"
echo "  export LD_LIBRARY_PATH=$DICE_BIN_PATH:\$LD_LIBRARY_PATH"
