tp=$1
oseqlen=$2

echo "Benchmarking FT T5-11B with TP=${tp}, output seq len=${oseqlen}"

echo "Batch Size 1, T5-11B, output_seq_len=${oseqlen}, TP=${tp}" >> t5_11b_${tp}gpu_oseqlen${oseqlen}.txt
mpirun -n ${tp} --allow-run-as-root python3 T5_perf_benchmark.py --batch_size 1 --data_type fp16 --beam_width 1 --seq_len ${oseqlen} --input_seq_len 64 --test_time 1 --len_penalty 2.0 --model t5-11b --warmup_iterations 2 --iterations 10 --duration 0 --tensor_para_size ${tp} | tee  >(tail -n 1 >> t5_11b_${tp}gpu_oseqlen${oseqlen}.txt) &&

echo "Batch Size 2, T5-11B, output_seq_len=${oseqlen}, TP=${tp}" >> t5_11b_${tp}gpu_oseqlen${oseqlen}.txt
mpirun -n ${tp} --allow-run-as-root python3 T5_perf_benchmark.py --batch_size 2 --data_type fp16 --beam_width 1 --seq_len ${oseqlen} --input_seq_len 64 --test_time 1 --len_penalty 2.0 --model t5-11b --warmup_iterations 2 --iterations 10 --duration 0 --tensor_para_size ${tp} | tee  >(tail -n 1 >> t5_11b_${tp}gpu_oseqlen${oseqlen}.txt) &&

echo "Batch Size 4, T5-11B, output_seq_len=${oseqlen}, TP=${tp}" >> t5_11b_${tp}gpu_oseqlen${oseqlen}.txt
mpirun -n ${tp} --allow-run-as-root python3 T5_perf_benchmark.py --batch_size 4 --data_type fp16 --beam_width 1 --seq_len ${oseqlen} --input_seq_len 64 --test_time 1 --len_penalty 2.0 --model t5-11b --warmup_iterations 2 --iterations 10 --duration 0 --tensor_para_size ${tp} | tee  >(tail -n 1 >> t5_11b_${tp}gpu_oseqlen${oseqlen}.txt) &&

echo "Batch Size 8, T5-11B, output_seq_len=${oseqlen}, TP=${tp}" >> t5_11b_${tp}gpu_oseqlen${oseqlen}.txt
mpirun -n ${tp} --allow-run-as-root python3 T5_perf_benchmark.py --batch_size 8 --data_type fp16 --beam_width 1 --seq_len ${oseqlen} --input_seq_len 64 --test_time 1 --len_penalty 2.0 --model t5-11b --warmup_iterations 2 --iterations 10 --duration 0 --tensor_para_size ${tp} | tee  >(tail -n 1 >> t5_11b_${tp}gpu_oseqlen${oseqlen}.txt) &&

echo "Batch Size 32, T5-11B, output_seq_len=${oseqlen}, TP=${tp}" >> t5_11b_${tp}gpu_oseqlen${oseqlen}.txt
mpirun -n ${tp} --allow-run-as-root python3 T5_perf_benchmark.py --batch_size 32 --data_type fp16 --beam_width 1 --seq_len ${oseqlen} --input_seq_len 64 --test_time 1 --len_penalty 2.0 --model t5-11b --warmup_iterations 2 --iterations 10 --duration 0 --tensor_para_size ${tp} | tee  >(tail -n 1 >> t5_11b_${tp}gpu_oseqlen${oseqlen}.txt) &&

echo "Finished benchmarking.."
echo "Printing results..."

cat t5_11b_${tp}gpu_oseqlen${oseqlen}.txt
