#!/bin/bash

# while-read2: read lines from a file

sort -k 1,1 -k 2n distros.txt | while read -r distro version release; do
	printf "Distro: %s\tVersion: %s\tReleased: %s\n" \
		"$distro" \
		"$version" \
		"$release"
done

