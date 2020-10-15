#!/bin/sh
run_scripts/fitrec/ds/iid/lr0.1_le1_bs2.sh \
&& run_scripts/fitrec/ds/iid/lr0.1_le1_bs4.sh \
&& run_scripts/fitrec/ds/iid/lr0.1_le2_bs2.sh \
&& run_scripts/fitrec/ds/iid/lr0.1_le2_bs4.sh \
&& run_scripts/fitrec/ds/iid/lr0.5_le1_bs2.sh \
&& run_scripts/fitrec/ds/iid/lr0.5_le1_bs4.sh \
&& run_scripts/fitrec/ds/iid/lr0.5_le2_bs2.sh \
&& run_scripts/fitrec/ds/iid/lr0.5_le2_bs4.sh \
&& run_scripts/fitrec/ds/non/lr0.1_le1_bs2.sh \
&& run_scripts/fitrec/ds/non/lr0.1_le1_bs4.sh \
&& run_scripts/fitrec/ds/non/lr0.1_le2_bs2.sh \
&& run_scripts/fitrec/ds/non/lr0.1_le2_bs4.sh \
&& run_scripts/fitrec/ds/non/lr0.5_le1_bs2.sh \
&& run_scripts/fitrec/ds/non/lr0.5_le1_bs4.sh \
&& run_scripts/fitrec/ds/non/lr0.5_le2_bs2.sh \
&& run_scripts/fitrec/ds/non/lr0.5_le2_bs4.sh