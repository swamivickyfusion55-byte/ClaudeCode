#!/bin/bash
# run a command inside the 2-vCPU CFS quota cgroup (all 4 cores visible, like a Hugging Face container)
echo $$ > /sys/fs/cgroup/cpu/phx2/cgroup.procs
exec "$@"
