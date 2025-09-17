#!/bin/bash

# test-integer2a: evaluate the value of an integer.

INT=-5

if [[ "$INT" =~ ^-?[0-9]+$ ]]; then
	if ((INT == 0)); then
		echo "INT is zero."
	else
		if ((INT < 0)); then
			echo "INT is negative."
		else
			echo "INT is positive."
		fi
		if (( ((INT % 2)) == 0)); then
			echo "INT is even."
		else
			echo "INT is odd."
		fi
	fi
else
	echo "INT is not an integer." >&2
	exit 1
fi
