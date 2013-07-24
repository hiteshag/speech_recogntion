#!/bin/bash

(
  script=$1
  shift
  echo -n 'args = {';
  for argument in $*; do
    echo -n " '$argument'"
  done;
  echo ' };';
  echo "$script";
  echo 'exit';
) | /usr/local/MATLAB/R2012a/bin/matlab -nosplash -nodesktop
