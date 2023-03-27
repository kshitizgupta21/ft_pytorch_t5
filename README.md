
1. Get FT
```
bash get_ft.sh
```

2. Run NGC PyTorch container
```
bash pytorch.sh
```

3. Build FT for H100
```
bash build_ft.sh
```

4. Install Requirements
```
cd ../..
pip install -r FasterTransformer/examples/pytorch/t5/requirement.txt
pip install accelerate
```

5. Run benchmark script for T5-11B


To run with 8-GPU TP, T5-11B, output_seq_len=32 use this

```
bash benchmark_t5_11b.sh 8 32
```

First argument is TP size and second argument is output sequence length.

You will see something like this at end of running the script
```
Batch Size 1, T5-11B, output_seq_len=1, TP=1
[INFO] ft-beamsearch translates 10 batches taking 0.38 sec to translate 10 tokens (38.1438 ms per batch), 26 tokens/sec.
Batch Size 2, T5-11B, output_seq_len=1, TP=1
[INFO] ft-beamsearch translates 10 batches taking 0.44 sec to translate 20 tokens (44.0612 ms per batch), 45 tokens/sec.
Batch Size 4, T5-11B, output_seq_len=1, TP=1
[INFO] ft-beamsearch translates 10 batches taking 0.59 sec to translate 40 tokens (59.4714 ms per batch), 67 tokens/sec.
Batch Size 8, T5-11B, output_seq_len=1, TP=1
[INFO] ft-beamsearch translates 10 batches taking 0.88 sec to translate 80 tokens (87.8921 ms per batch), 91 tokens/sec.
Batch Size 32, T5-11B, output_seq_len=1, TP=1
[INFO] ft-beamsearch translates 10 batches taking 2.57 sec to translate 320 tokens (256.5925 ms per batch), 125 tokens/sec.
```

Here ms per batch is latency and tokens/sec is throughput

If you are also are interested in output_seq_len=1 (aka time to first token) so you can run 

```
bash benchmark_t5_11b.sh 8 1
```




