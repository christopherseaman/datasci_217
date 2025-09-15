#!/bin/bash

# trap-demo3 - demonstrate ERR and EXIT signal handling

trap "echo \"There is an error.\"" ERR
trap "echo \"The program has ended.\"" EXIT

echox "Running..."

read -r -p "Say something... " something
echo "$something"
