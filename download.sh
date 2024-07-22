#!/bin/bash

rm listings.csv
URL="https://data.insideairbnb.com/united-states/ny/new-york-city/2024-07-05/data/listings.csv.gz"
DEST_FILE="listings.csv.gz"
wget $URL -O $DEST_FILE
gunzip $DEST_FILE


rm reviews.csv
URL="https://data.insideairbnb.com/united-states/ny/new-york-city/2024-07-05/data/reviews.csv.gz"
DEST_FILE="reviews.csv.gz"
wget $URL -O $DEST_FILE
gunzip $DEST_FILE


