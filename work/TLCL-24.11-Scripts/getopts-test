#!/bin/bash

# getopts-test: process command line options using getopts

PROGNAME="$(basename "$0")"
interactive=
filename=

usage () {
    echo "$PROGNAME: usage: $PROGNAME [-f file | -i]"
    return
}

while getopts :f:ih opt; do
    case "$opt" in
        f)  filename="$OPTARG" ;;
        i)  interactive=1 ;;
        h)  usage ;;
        \?) echo "option '$OPTARG' invalid" ;;
        :)  echo "option '$OPTARG' missing argument";;
    esac
done
echo "interactive = '$interactive' filename = '$filename'"
