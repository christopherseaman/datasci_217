#!/bin/bash

# while-menu2: a menu driven system information program

DELAY=3 # Number of seconds to display results

while true; do
	clear
	cat <<- _EOF_
		Please Select:

		1. Display System Information
		2. Display Disk Space
		3. Display Home Space Utilization
		0. Quit

_EOF_
	read -r -p "Enter selection [0-3] > "

	if [[ "$REPLY" =~ ^[0-3]$ ]]; then
		if [[ "$REPLY" == 1 ]]; then
			echo "Hostname: $HOSTNAME"
			uptime
			sleep "$DELAY"
			continue
		fi
		if [[ "$REPLY" == 2 ]]; then
			df -h
			sleep $DELAY
			continue
		fi
		if [[ "$REPLY" == 3 ]]; then
			if [[ "$(id -u)" -eq 0 ]]; then
				echo "Home Space Utilization (All Users)"
				du -sh /home/*
			else
				echo "Home Space Utilization ($USER)"
				du -sh "$HOME"
			fi
			sleep "$DELAY"
			continue
		fi
		if [[ "$REPLY" == 0 ]]; then
			break
		fi
	else
		echo "Invalid entry."
		sleep "$DELAY"
	fi
done
echo "Program terminated."

