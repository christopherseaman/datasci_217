#!/bin/bash

# local-vars: script to demonstrate local variables

foo=0	# global variable foo

funct_1 () {

	local foo	# variable foo local to funct_1

	foo=1
	echo "funct_1: foo = $foo"
}

funct_2 () {

	local foo	# variable foo local to funct_2

	foo=2
	echo "funct_2: foo = $foo"
}

echo "global:  foo = $foo"
funct_1
echo "global:  foo = $foo"
funct_2
echo "global:  foo = $foo"
