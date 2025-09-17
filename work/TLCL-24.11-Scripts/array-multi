#!/bin/bash

# array-multi - simulate a multi-dimensional array

declare -A multi_array

# Load array with a sequence of numbers
counter=1
for row in {1..10}; do
  for col in {1..5}; do
    address="$row, $col"
    multi_array["$address"]=$counter
    ((counter++))
  done
done

# Output array contents
for row in {1..10}; do
  for col in {1..5}; do
    address="$row, $col"
    echo -ne "${multi_array["$address"]}" "\t"
  done
  echo
done

