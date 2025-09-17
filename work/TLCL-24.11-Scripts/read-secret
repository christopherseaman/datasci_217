#!/bin/bash

# read-secret: input a secret pass phrase

if read -r -t 10 -sp "Enter secret pass phrase > " secret_pass; then
	echo -e "\nSecret pass phrase = '$secret_pass'"
else
	echo -e "\nInput timed out" 2>&1
	exit 1
fi

