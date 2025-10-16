#!/bin/bash

# create dir just in case
mkdir -p .local

awk 'BEGIN {print "## CHANGELOG"} 
     /^### V-/ {if (found) exit; found=1; next} 
     found && NF {print}' CHANGELOG.md > .local/release.md
