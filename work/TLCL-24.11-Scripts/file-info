#!/bin/bash

# file_info: simple file information program

PROGNAME="$(basename "$0")"

if [[ -e "$1" ]]; then
	echo -e "\nFile Type:"
	file "$1"
	echo -e "\nFile Status:"
	stat "$1"
else
	echo "$PROGNAME: usage: $PROGNAME file" >&2
	exit 1
fi

