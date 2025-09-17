#!/bin/bash

# average - script to calculate the average of a series of numbers

# handle cmd line option
if [[ $1 ]]; then
	case $1 in
		-s|--scale)	scale=$2 ;;
		*)		echo "usage: average [-s scale]" >&2
				exit 1 ;;
	esac
fi

# construct instruction stream for bc
c=0
{	echo "t = 0; scale = 2"
	[[ $scale ]] && echo "scale = $scale"
	while read -r value; do

		# only process valid numbers
		if [[ $value =~ ^[-+]?[0-9]*\.?[0-9]+$ ]]; then
			echo "t += $value"
			((++c))
		fi
	done

	# make sure we don't divide by zero
	((c)) && echo "t / $c"
} | bc 
