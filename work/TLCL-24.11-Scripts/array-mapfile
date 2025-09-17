#!/bin/bash

# array-mapfile - demonstrate mapfile builtin

DICTIONARY=/usr/share/dict/words
WORDLIST=~/wordlist.txt
declare -a words

# create filtered word list
grep -v \' < "$DICTIONARY" \
    | grep -v "[[:upper:]]" \
    | shuf > "$WORDLIST"

# read WORDLIST into array
mapfile -t -n 32767 words < "$WORDLIST"

# create four word passphrase
while [[ -z $REPLY ]]; do
    echo "${words[$RANDOM]}" \
         "${words[$RANDOM]}" \
         "${words[$RANDOM]}" \
         "${words[$RANDOM]}"
    echo
    read -r -p "Enter to continue, q to quit > "
    echo
done

