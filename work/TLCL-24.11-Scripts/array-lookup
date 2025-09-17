#!/bin/bash

# array-lookup - demonstrate lookup using associative array

declare -A cmds

# fill array with commands and file sizes
cd /usr/bin || exit 1
echo "Loading commands..."
for i in ./*; do
  cmds["$i"]=$(stat -c "%s" "$i")
done
echo "${#cmds[@]} commands loaded"

# perform lookup 
while true; do
  read -r -p "Enter command (empty to quit) -> "
  [[ -z $REPLY ]] && break
  if [[ -x $REPLY ]]; then
    echo "$REPLY" "${cmds[./$REPLY]}" "bytes"
  else
    echo "No such command '$REPLY'."
  fi
done

