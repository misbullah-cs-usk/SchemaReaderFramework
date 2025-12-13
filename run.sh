#!/bin/bash

#--input-data /home/alim/Advanced-DB/github-event/${f}/github_events_v2_${f}.jsonl \
#--input-data /home/alim/Advanced-DB/sensor_data_more_features/sensor_data_more_features_${f}_valid.jsonl \
for f in 10k 25k 50k 100k 500k 1m 2m 5m; do
  python main.py --mode benchmark \
    --chunk-size 10000 \
    --input-data /home/alim/Advanced-DB/github-event/${f}/github_events_v2_${f}.jsonl \
    --num-workers 1 \
    --repeat 5 \
    --formats jsonl csv parquet avro feather \
    --output-dir output/
done
