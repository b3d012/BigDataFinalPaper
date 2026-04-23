# Edge-IIoT Drift Detection Summary

- Mode: demo
- Reference batch: dataset_train_benign_split
- Target batch: demo_replay_rows
- Reference rows: 19441
- Target rows: 170575
- Feature count: 32
- Overall severity: high
- Drift flag: True
- Drift rule: max PSI < 0.10 => low, 0.10 <= max PSI < 0.25 => moderate, max PSI >= 0.25 => high

## Top Drifted Features
- num__tcp.ack_raw: PSI=12.572593, JS=0.525553
- num__tcp.checksum: PSI=12.432733, JS=0.525486
- num__tcp.seq: PSI=11.780617, JS=0.471940
- num__tcp.ack: PSI=11.530123, JS=0.466246
- num__tcp.len: PSI=3.915813, JS=0.129131
- num__tcp.flags: PSI=3.238909, JS=0.333186
- num__mqtt.hdrflags: PSI=2.429372, JS=0.077438
- num__mqtt.msgtype: PSI=2.429372, JS=0.077438
- num__tcp.flags.ack: PSI=1.835207, JS=0.055701
- num__mqtt.len: PSI=1.187968, JS=0.036735
- num__tcp.connection.syn: PSI=1.183899, JS=0.127272
- num__tcp.connection.rst: PSI=0.563503, JS=0.066163
- num__mqtt.topic_len: PSI=0.563164, JS=0.018235
- num__mqtt.conflag.cleansess: PSI=0.548416, JS=0.017791
- num__mqtt.conflags: PSI=0.548416, JS=0.017791

## Notes
- PSI is computed on the transformed feature space produced by the saved classifier contract.