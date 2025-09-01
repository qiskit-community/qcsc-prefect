#!/bin/bash
set -euo pipefail

# Load modules (Intel compiler, MPI, Boost with MPI, and Parallel HDF5)
module load intel/2023.2.0
module load impi/2021.10.0
module load phdf5/1.14.3

# Start with a fresh dice_solver dir
rm -rf dice_solver/ && mkdir dice_solver
cd dice_solver

BOOST_ROOT="$(pwd)/boost_1_85_0"
DICE_ROOT="$(pwd)/Dice"
BIN_DIR="$(pwd)/bin"
OBJ_DIR="$DICE_ROOT/obj/SHCI"

# Get Boost
wget https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.gz
tar -xzf boost_1_85_0.tar.gz
rm boost_1_85_0.tar.gz

# Build boost
cd "$BOOST_ROOT"
./bootstrap.sh --with-libraries=serialization,mpi
echo 'using intel-linux : icpx : icpx : <compileflags>"-axSAPPHIRERAPIDS,CORE-AVX512 -qopenmp -diag-disable=10430" ;' > user-config.jam
echo 'using mpi : mpiicx : <compileflags>"-axSAPPHIRERAPIDS,CORE-AVX512 -qopenmp -diag-disable=10430" ;' >> user-config.jam
./b2 toolset=intel-linux --user-config=user-config.jam -j$(nproc)

# Clone Dice default branch
git clone https://github.com/caleb-johnson/Dice.git "$DICE_ROOT"
cd "$DICE_ROOT"

# Build DICE
SHCI_SRC_FILES=(
    SHCIbasics.cpp 
    Determinants.cpp 
    integral.cpp 
    input.cpp 
    Davidson.cpp
    SHCIgetdeterminants.cpp 
    SHCIsampledeterminants.cpp 
    SHCIrdm.cpp 
    SHCISortMpiUtils.cpp
    SHCImakeHamiltonian.cpp 
    SHCIshm.cpp 
    LCC.cpp 
    symmetry.cpp 
    OccRestrictions.cpp
    cdfci.cpp 
    SHCI.cpp
)

FLAGS="-std=c++14 -O3 -g -w -fPIC -qopenmp \
       -I. \
       -I./eigen/ \
       -I./SHCI \
       -I$BOOST_ROOT \
       -I$HDF5_INC \
       -axSAPPHIRERAPIDS,CORE-AVX512" # Recommended optimization option for Miyabi-C node

LINK_FLAGS="-L$BOOST_ROOT/stage/lib -lboost_mpi -lboost_serialization"

VERSION_FLAGS="-DGIT_HASH=\"$(git rev-parse HEAD)\" \
               -DGIT_BRANCH=\"$(git branch --show-current)\" \
               -DCOMPILE_TIME=\"$(date '+%Y-%m-%d_%H-%M-%S')\""

# Compile each SHCI source file
OBJ_Dice=""
for src in "${SHCI_SRC_FILES[@]}"; do
    obj="obj/SHCI/${src%.cpp}.o"
    OBJ_Dice+=" $obj"
    mpiicpx $FLAGS $VERSION_FLAGS -c SHCI/$src -o $obj
done

mpiicpx $FLAGS -o bin/Dice $OBJ_Dice $LINK_FLAGS

# Put the runtime libraries in the package
mkdir -p $BIN_DIR
cp $DICE_ROOT/bin/Dice $BIN_DIR
cp $BOOST_ROOT/stage/lib/*.so* $BIN_DIR