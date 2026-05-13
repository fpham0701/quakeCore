# Live Quake Co-Traversal on Perlmutter

## One-time setup on the login node

```bash
git clone <quakecore repo> ~/quakeCore
cd ~/quakeCore
module load PrgEnv-gnu cudatoolkit cmake
cmake -S . -B build \
  -DQUAKECORE_BUILD_CPU_OPT=ON -DQUAKECORE_BUILD_GPU_OPT=ON \
  -DQUAKECORE_BUILD_BENCH=ON   -DQUAKECORE_BUILD_LIVE=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j 32 --target quakecore_live
```

Copy the BSP file the workstation will use to `$SCRATCH/maps/<map>.bsp` (the sha256 must match exactly).

## Per-session

1. Allocate an A100 interactively:
   ```
   salloc -N1 -G1 -t 02:00:00 -C gpu -A <project>
   ```
   Note the assigned hostname, e.g. `nid006789`.

2. On the compute node, start the sidecar listening on TCP:
   ```
   srun --ntasks=1 ~/quakeCore/build/quakecore_live \
        --transport tcp-listen:5000 \
        --csv $SCRATCH/runs/$SLURM_JOB_ID.csv \
        --threads 64 --block-size 256
   ```

3. From the workstation, open the SSH tunnel:
   ```
   ssh -L 5000:nid006789:5000 perlmutter.nersc.gov
   ```
   Keep this terminal open for the duration of play.

4. Launch the patched Ironwail on the workstation:
   ```
   ./build/ironwail/ironwail -basedir ~/quake +map start \
     --qcfp-transport tcp:127.0.0.1:5000 --qcfp-handshake-once
   ```

5. Quit Ironwail when done. The sidecar will see EOF, flush the CSV, and exit. Pull the CSV back:
   ```
   scp perlmutter.nersc.gov:$SCRATCH/runs/<job>.csv .
   ```

## Notes

- The BSP at the path given in the workstation's handshake (e.g. `id1/maps/start.bsp`) is *workstation-local*. The sidecar will reject the handshake because that path doesn't exist on Perlmutter — *or* it'll hash a different file with the same path. Two workarounds:
  - (Recommended) Run with `--map-override $SCRATCH/maps/start.bsp` once that flag is implemented.
  - (V1) Symlink: on Perlmutter, `ln -s $SCRATCH/maps/start.bsp id1/maps/start.bsp` inside a directory you `cd` into before launching the sidecar.

- Wall-time: interactive Slurm jobs are typically capped at 8h. Renew with another `salloc` for long sessions.
- Tunnel drops: Ironwail logs "transport disconnected" and continues playing. Restart the tunnel + sidecar; restart Ironwail to re-handshake.
