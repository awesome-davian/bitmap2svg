#!/bin/bash
node circle.js 0
for i in `seq 1 19`; do
  num="$(find ./bitmap -type f | wc -l)"
  temp=$((100 * $i))
  if [ ${num} -eq ${temp} ];then
    node circle.js $i;
    echo ==================================$i========================================
  fi
done

