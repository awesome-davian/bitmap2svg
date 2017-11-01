#!/bin/bash
node circle.js 0
for i in `seq 1 1000`; do
  echo 123!!!!
  num="$(find ./bitmap -type f | wc -l)"
  temp=$((100 * $i))
  if [ ${num} -eq ${temp} ];then
    node circle.js $i;
    echo ==================================$i========================================
  fi
done

