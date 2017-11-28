#!/bin/bash
node barchart.js 0
for i in `seq 0 999`; do
  num="$(find ./bitmap -type f | wc -l)"
  temp=$((100 * $i))
  if [ ${num} -eq ${temp} ];then
    node barchart.js $i;
    echo ==================================$i========================================
  fi
done

