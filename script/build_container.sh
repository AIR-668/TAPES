#!/bin/bash
set -e
echo "[INFO] Building TAPES container with fakeroot..."
apptainer build --fakeroot tapes.sif apptainer/tapes.def
echo "[INFO] Done. Container image is tapes.sif"