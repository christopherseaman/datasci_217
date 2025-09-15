#!/bin/bash

# hours : script to count files by modification time

usage () {
	echo "usage: ${0##*/} directory" >&2
}

# Check that argument is a directory
if [[ ! -d "$1" ]]; then
	usage
	exit 1
fi

# Initialize array
for i in {0..23}; do hours[i]=0; done 

# Collect data
for i in $(stat -c %y "$1"/* | cut -c 12-13); do
	j="${i/#0}"
	((++hours[j]))
	((++count))
done

# Display data
echo -e "Hour\tFiles\tHour\tFiles"
echo -e "----\t-----\t----\t-----"
for i in {0..11}; do
	j=$((i + 12))
	printf "%02d\t%d\t%02d\t%d\n" \
		"$i" \
		"${hours[i]}" \
		"$j" \
		"${hours[j]}"
done
printf "\nTotal files = %d\n" "$count"

