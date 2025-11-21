import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

with open('../results/Parallel/output.json') as f:
    data = json.load(f)

df = pd.json_normalize(data['results'])
hw = data['hardware']

# Peak FLOP/s (esempio Xeon Gold: 96 core × 2.3 GHz × 8 FMA = ~1.8 TFLOP/s)
peak_flops = hw['total_cores'] * hw['cpu_mhz'] / 1000 * 8  # in GFLOP/s

# Peak Bandwidth (esempio: 400 GB/s)
peak_bw = 400  # vai dal datasheet

# Roofline plot
fig, ax = plt.subplots()
ai_range = np.logspace(-2, 1, 100)
compute_roof = np.ones_like(ai_range) * peak_flops
bandwidth_roof = ai_range * peak_bw

ax.loglog(ai_range, compute_roof, 'r-', label='Compute Roof')
ax.loglog(ai_range, bandwidth_roof, 'b-', label='Memory Roof')
ax.scatter(df['statistics_90.arithmetic_intensity'],
           df['statistics_90.GFLOP/s'], c='g', s=50, marker='o')
ax.set_xlabel('Arithmetic Intensity [FLOPs/byte]')
ax.set_ylabel('GFLOP/s')
ax.legend()
plt.savefig('roofline.png')
plt.show()
