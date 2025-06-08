import re

def parse_log_file(log_path):
    step_times = []
    gpu_memories = []

    # Regex pattern to match lines like:
    # [Epoch 0] Average Step Time: 9.861s | Average GPU Memory: 37848.6 GB
    pattern = re.compile(r"\[Epoch \d+\] Average Step Time: ([\d.]+)s \| Average GPU Memory: ([\d.]+) GB")

    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                step_time = float(match.group(1))
                gpu_mem = float(match.group(2))
                step_times.append(step_time)
                gpu_memories.append(gpu_mem)

    if step_times:
        avg_step_time = sum(step_times) / len(step_times)
        avg_gpu_mem = sum(gpu_memories) / len(gpu_memories)

        print(f"📊 Parsed {len(step_times)} epochs")
        print(f"✅ Average Step Time: {avg_step_time:.3f} s")
        print(f"✅ Average GPU Memory: {avg_gpu_mem:.1f} MB")
    else:
        print("⚠️ No matching logs found in the file.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python parse_training_log.py path_to_log.txt")
    else:
        parse_log_file(sys.argv[1])