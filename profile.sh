!#/bin/bash

cargo flamegraph --example $1 --release -c "record --call-graph fp"

rm -rf perf.data*