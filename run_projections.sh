#!/bin/bash

for i in {145..247}
do
    python -u snap_loop_over_halos.py $i > $i.log 2>&1 &
    if [ $(($i % 16)) == 0 ]
    then
	wait
	echo "waiting"
    fi
done
