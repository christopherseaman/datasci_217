#!/bin/bash

# read-ifs: read fields from a file

FILE=/etc/passwd

read -r -p "Enter a user name > " user_name

file_info="$(grep "^$user_name:" $FILE)"

if [ -n "$file_info" ]; then
	IFS=":" read user pw uid gid name home shell <<< "$file_info"
	echo "User =      '$user'"
	echo "UID =       '$uid'"
	echo "GID =       '$gid'"
	echo "Full Name = '$name'"
	echo "Home Dir. = '$home'"
	echo "Shell =     '$shell'" 
else
	echo "No such user '$user_name'" 2>&1
	exit 1
fi
