#!/bin/bash
# change the following three lines to the values you need
product=MYD16A2
collection=6
year=2005

until wget -m  https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/$collection/$product/$year/
do
 echo "Trying again in 10 seconds..."
 sleep 10
done
