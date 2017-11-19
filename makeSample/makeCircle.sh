#!/bin/bash
node random_circleAndrect_3.js  0
for i in `seq 0 99`; do
  num="$(find ./bitmap -type f | wc -l)"
  temp=$((100 * $i))
  if [ ${num} -eq ${temp} ];then
    node random_circleAndrect_3.js $i;
    echo ==================================$i========================================
  fi
done

