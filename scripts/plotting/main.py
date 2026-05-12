import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

# Input data
csv_data = """block_size,16,32,60000,7,quake_map_source-master__bsp__e1m1.bsp,baseline,60000,0.0541379,1.10828e+06,60000,0,60000,0
block_size,16,32,60000,7,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,60000,0.00425607,1.40975e+07,60000,0,60000,0
block_size,16,32,60000,7,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,60000,0.467181,128430,60000,0,60000,0
block_size,16,32,60000,13,quake_map_source-master__bsp__e1m1.bsp,baseline,60000,0.0530323,1.13139e+06,60000,0,60000,0
block_size,16,32,60000,13,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,60000,0.0050652,1.18455e+07,60000,0,60000,0
block_size,16,32,60000,13,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,60000,0.313096,191635,60000,0,60000,0
block_size,16,32,60000,42,quake_map_source-master__bsp__e1m1.bsp,baseline,60000,0.0529065,1.13408e+06,60000,0,60000,0
block_size,16,32,60000,42,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,60000,0.00459637,1.30538e+07,60000,0,60000,0
block_size,16,32,60000,42,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,60000,0.297141,201925,60000,0,60000,0
block_size,16,64,60000,7,quake_map_source-master__bsp__e1m1.bsp,baseline,60000,0.0526833,1.13888e+06,60000,0,60000,0
block_size,16,64,60000,7,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,60000,0.00491732,1.22018e+07,60000,0,60000,0
block_size,16,64,60000,7,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,60000,0.296484,202372,60000,0,60000,0
block_size,16,64,60000,13,quake_map_source-master__bsp__e1m1.bsp,baseline,60000,0.056603,1.06002e+06,60000,0,60000,0
block_size,16,64,60000,13,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,60000,0.0051357,1.16829e+07,60000,0,60000,0
block_size,16,64,60000,13,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,60000,0.26899,223057,60000,0,60000,0
block_size,16,64,60000,42,quake_map_source-master__bsp__e1m1.bsp,baseline,60000,0.0538318,1.11458e+06,60000,0,60000,0
block_size,16,64,60000,42,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,60000,0.00492843,1.21743e+07,60000,0,60000,0
block_size,16,64,60000,42,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,60000,0.28276,212194,60000,0,60000,0
block_size,16,128,60000,7,quake_map_source-master__bsp__e1m1.bsp,baseline,60000,0.0537855,1.11554e+06,60000,0,60000,0
block_size,16,128,60000,7,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,60000,0.00451335,1.32939e+07,60000,0,60000,0
block_size,16,128,60000,7,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,60000,0.307694,194999,60000,0,60000,0
block_size,16,128,60000,13,quake_map_source-master__bsp__e1m1.bsp,baseline,60000,0.0535273,1.12092e+06,60000,0,60000,0
block_size,16,128,60000,13,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,60000,0.00506991,1.18345e+07,60000,0,60000,0
block_size,16,128,60000,13,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,60000,0.303034,197997,60000,0,60000,0
block_size,16,128,60000,42,quake_map_source-master__bsp__e1m1.bsp,baseline,60000,0.0546853,1.09719e+06,60000,0,60000,0
block_size,16,128,60000,42,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,60000,0.00466752,1.28548e+07,60000,0,60000,0
block_size,16,128,60000,42,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,60000,0.282657,212272,60000,0,60000,0
block_size,16,256,60000,7,quake_map_source-master__bsp__e1m1.bsp,baseline,60000,0.0543512,1.10393e+06,60000,0,60000,0
block_size,16,256,60000,7,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,60000,0.00501318,1.19685e+07,60000,0,60000,0
block_size,16,256,60000,7,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,60000,0.274892,218268,60000,0,60000,0
block_size,16,256,60000,13,quake_map_source-master__bsp__e1m1.bsp,baseline,60000,0.0545814,1.09928e+06,60000,0,60000,0
block_size,16,256,60000,13,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,60000,0.00453556,1.32288e+07,60000,0,60000,0
block_size,16,256,60000,13,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,60000,0.300192,199872,60000,0,60000,0
block_size,16,256,60000,42,quake_map_source-master__bsp__e1m1.bsp,baseline,60000,0.0537375,1.11654e+06,60000,0,60000,0
block_size,16,256,60000,42,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,60000,0.00440701,1.36147e+07,60000,0,60000,0
block_size,16,256,60000,42,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,60000,0.314733,190638,60000,0,60000,0
block_size,16,512,60000,7,quake_map_source-master__bsp__e1m1.bsp,baseline,60000,0.0545337,1.10024e+06,60000,0,60000,0
block_size,16,512,60000,7,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,60000,0.00465454,1.28906e+07,60000,0,60000,0
block_size,16,512,60000,7,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,60000,0.282553,212349,60000,0,60000,0
block_size,16,512,60000,13,quake_map_source-master__bsp__e1m1.bsp,baseline,60000,0.0545612,1.09968e+06,60000,0,60000,0
block_size,16,512,60000,13,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,60000,0.00446926,1.3425e+07,60000,0,60000,0
block_size,16,512,60000,13,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,60000,0.303591,197634,60000,0,60000,0
block_size,16,512,60000,42,quake_map_source-master__bsp__e1m1.bsp,baseline,60000,0.0552511,1.08595e+06,60000,0,60000,0
block_size,16,512,60000,42,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,60000,0.00468851,1.27972e+07,60000,0,60000,0
block_size,16,512,60000,42,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,60000,0.304101,197303,60000,0,60000,0
block_size,16,1024,60000,7,quake_map_source-master__bsp__e1m1.bsp,baseline,60000,0.0532567,1.12662e+06,60000,0,60000,0
block_size,16,1024,60000,7,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,60000,0.00488406,1.22849e+07,60000,0,60000,0
block_size,16,1024,60000,7,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,60000,0.313292,191515,60000,0,60000,0
block_size,16,1024,60000,13,quake_map_source-master__bsp__e1m1.bsp,baseline,60000,0.0538445,1.11432e+06,60000,0,60000,0
block_size,16,1024,60000,13,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,60000,0.00444152,1.35089e+07,60000,0,60000,0
block_size,16,1024,60000,13,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,60000,0.279691,214522,60000,0,60000,0
block_size,16,1024,60000,42,quake_map_source-master__bsp__e1m1.bsp,baseline,60000,0.0540502,1.11008e+06,60000,0,60000,0
block_size,16,1024,60000,42,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,60000,0.00448097,1.33899e+07,60000,0,60000,0
block_size,16,1024,60000,42,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,60000,0.297812,201469,60000,0,60000,0
frames,16,256,500,7,quake_map_source-master__bsp__e1m1.bsp,baseline,500,0.000489197,1.02208e+06,500,0,500,0
frames,16,256,500,7,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,500,0.000783136,638459,500,0,500,0
frames,16,256,500,7,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,500,0.328054,1524.14,500,0,500,0
frames,16,256,500,13,quake_map_source-master__bsp__e1m1.bsp,baseline,500,0.000476713,1.04885e+06,500,0,500,0
frames,16,256,500,13,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,500,0.000742497,673403,500,0,500,0
frames,16,256,500,13,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,500,0.332618,1503.22,500,0,500,0
frames,16,256,500,42,quake_map_source-master__bsp__e1m1.bsp,baseline,500,0.000438228,1.14096e+06,500,0,500,0
frames,16,256,500,42,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,500,0.000993363,503341,500,0,500,0
frames,16,256,500,42,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,500,0.324415,1541.24,500,0,500,0
frames,16,256,1000,7,quake_map_source-master__bsp__e1m1.bsp,baseline,1000,0.000870396,1.1489e+06,1000,0,1000,0
frames,16,256,1000,7,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,1000,0.000799047,1.25149e+06,1000,0,1000,0
frames,16,256,1000,7,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,1000,0.337297,2964.75,1000,0,1000,0
frames,16,256,1000,13,quake_map_source-master__bsp__e1m1.bsp,baseline,1000,0.000865826,1.15497e+06,1000,0,1000,0
frames,16,256,1000,13,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,1000,0.000783757,1.27591e+06,1000,0,1000,0
frames,16,256,1000,13,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,1000,0.331673,3015.02,1000,0,1000,0
frames,16,256,1000,42,quake_map_source-master__bsp__e1m1.bsp,baseline,1000,0.000972573,1.0282e+06,1000,0,1000,0
frames,16,256,1000,42,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,1000,0.000783217,1.27679e+06,1000,0,1000,0
frames,16,256,1000,42,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,1000,0.339266,2947.54,1000,0,1000,0
frames,16,256,2000,7,quake_map_source-master__bsp__e1m1.bsp,baseline,2000,0.00178432,1.12088e+06,2000,0,2000,0
frames,16,256,2000,7,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,2000,0.0011967,1.67127e+06,2000,0,2000,0
frames,16,256,2000,7,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,2000,0.32861,6086.24,2000,0,2000,0
frames,16,256,2000,13,quake_map_source-master__bsp__e1m1.bsp,baseline,2000,0.00179872,1.1119e+06,2000,0,2000,0
frames,16,256,2000,13,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,2000,0.000902488,2.2161e+06,2000,0,2000,0
frames,16,256,2000,13,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,2000,0.336577,5942.18,2000,0,2000,0
frames,16,256,2000,42,quake_map_source-master__bsp__e1m1.bsp,baseline,2000,0.00176627,1.13233e+06,2000,0,2000,0
frames,16,256,2000,42,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,2000,0.000925562,2.16085e+06,2000,0,2000,0
frames,16,256,2000,42,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,2000,0.327101,6114.31,2000,0,2000,0
frames,16,256,5000,7,quake_map_source-master__bsp__e1m1.bsp,baseline,5000,0.00446994,1.11858e+06,5000,0,5000,0
frames,16,256,5000,7,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,5000,0.00124279,4.02322e+06,5000,0,5000,0
frames,16,256,5000,7,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,5000,0.329559,15171.8,5000,0,5000,0
frames,16,256,5000,13,quake_map_source-master__bsp__e1m1.bsp,baseline,5000,0.00451001,1.10864e+06,5000,0,5000,0
frames,16,256,5000,13,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,5000,0.00109639,4.56041e+06,5000,0,5000,0
frames,16,256,5000,13,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,5000,0.340135,14700,5000,0,5000,0
frames,16,256,5000,42,quake_map_source-master__bsp__e1m1.bsp,baseline,5000,0.00440527,1.135e+06,5000,0,5000,0
frames,16,256,5000,42,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,5000,0.00116051,4.30846e+06,5000,0,5000,0
frames,16,256,5000,42,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,5000,0.335539,14901.4,5000,0,5000,0
frames,16,256,10000,7,quake_map_source-master__bsp__e1m1.bsp,baseline,10000,0.00930707,1.07445e+06,10000,0,10000,0
frames,16,256,10000,7,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,10000,0.001516,6.59633e+06,10000,0,10000,0
frames,16,256,10000,7,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,10000,0.352782,28346.1,10000,0,10000,0
frames,16,256,10000,13,quake_map_source-master__bsp__e1m1.bsp,baseline,10000,0.00909487,1.09952e+06,10000,0,10000,0
frames,16,256,10000,13,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,10000,0.00194383,5.14447e+06,10000,0,10000,0
frames,16,256,10000,13,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,10000,0.34061,29359.1,10000,0,10000,0
frames,16,256,10000,42,quake_map_source-master__bsp__e1m1.bsp,baseline,10000,0.00886806,1.12764e+06,10000,0,10000,0
frames,16,256,10000,42,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,10000,0.00147235,6.79186e+06,10000,0,10000,0
frames,16,256,10000,42,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,10000,0.340925,29332,10000,0,10000,0
frames,16,256,20000,7,quake_map_source-master__bsp__e1m1.bsp,baseline,20000,0.0178684,1.11929e+06,20000,0,20000,0
frames,16,256,20000,7,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,20000,0.00233608,8.56134e+06,20000,0,20000,0
frames,16,256,20000,7,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,20000,0.402951,49633.9,20000,0,20000,0
frames,16,256,20000,13,quake_map_source-master__bsp__e1m1.bsp,baseline,20000,0.0182101,1.09829e+06,20000,0,20000,0
frames,16,256,20000,13,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,20000,0.00229298,8.72227e+06,20000,0,20000,0
frames,16,256,20000,13,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,20000,0.31553,63385.5,20000,0,20000,0
frames,16,256,20000,42,quake_map_source-master__bsp__e1m1.bsp,baseline,20000,0.0179312,1.11538e+06,20000,0,20000,0
frames,16,256,20000,42,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,20000,0.00229494,8.71481e+06,20000,0,20000,0
frames,16,256,20000,42,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,20000,0.334036,59873.8,20000,0,20000,0
frames,16,256,40000,7,quake_map_source-master__bsp__e1m1.bsp,baseline,40000,0.0364139,1.09848e+06,40000,0,40000,0
frames,16,256,40000,7,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,40000,0.00393371,1.01685e+07,40000,0,40000,0
frames,16,256,40000,7,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,40000,0.322399,124070,40000,0,40000,0
frames,16,256,40000,13,quake_map_source-master__bsp__e1m1.bsp,baseline,40000,0.0367818,1.0875e+06,40000,0,40000,0
frames,16,256,40000,13,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,40000,0.0033627,1.18952e+07,40000,0,40000,0
frames,16,256,40000,13,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,40000,0.291444,137248,40000,0,40000,0
frames,16,256,40000,42,quake_map_source-master__bsp__e1m1.bsp,baseline,40000,0.0385143,1.03858e+06,40000,0,40000,0
frames,16,256,40000,42,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,40000,0.00349978,1.14293e+07,40000,0,40000,0
frames,16,256,40000,42,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,40000,0.379584,105379,40000,0,40000,0
frames,16,256,60000,7,quake_map_source-master__bsp__e1m1.bsp,baseline,60000,0.0540951,1.10916e+06,60000,0,60000,0
frames,16,256,60000,7,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,60000,0.00502791,1.19334e+07,60000,0,60000,0
frames,16,256,60000,7,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,60000,0.319716,187666,60000,0,60000,0
frames,16,256,60000,13,quake_map_source-master__bsp__e1m1.bsp,baseline,60000,0.0537806,1.11564e+06,60000,0,60000,0
frames,16,256,60000,13,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,60000,0.00501769,1.19577e+07,60000,0,60000,0
frames,16,256,60000,13,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,60000,0.291767,205643,60000,0,60000,0
frames,16,256,60000,42,quake_map_source-master__bsp__e1m1.bsp,baseline,60000,0.05354,1.12066e+06,60000,0,60000,0
frames,16,256,60000,42,quake_map_source-master__bsp__e1m1.bsp,cpu_opt,60000,0.00471488,1.27257e+07,60000,0,60000,0
frames,16,256,60000,42,quake_map_source-master__bsp__e1m1.bsp,gpu_opt,60000,0.269816,222374,60000,0,60000,0"""

COLUMNS = [
    'test_type', 'threads', 'block_size', 'frames', 'seed', 'map',
    'version', 'frames_run', 'runtime', 'throughput',
    'visited_nodes', 'visited_leafs', 'culled_nodes', 'accepted_leafs',
]
df = pd.read_csv(io.StringIO(csv_data), header=None, names=COLUMNS)

def create_plots(df):
    df = df[df['version'].isin(['baseline', 'gpu_opt'])]

    frames_df = df[df['test_type'] == 'frames']
    bs_df = df[df['test_type'] == 'block_size']

    versions = ['baseline', 'gpu_opt']
    colors = {'baseline': '#1f77b4', 'gpu_opt': '#d62728'}
    markers = {'baseline': 'o', 'gpu_opt': 's'}

    def loglog_pair(ax_runtime, ax_throughput, sub_df, sweep_col, sweep_values, xlabel, title_suffix):
        sweep_arr = np.array(sweep_values, dtype=float)
        for v in versions:
            runtime_means = np.array([sub_df[(sub_df['version'] == v) & (sub_df[sweep_col] == x)]['runtime'].mean()
                                      for x in sweep_values])
            throughput_means = np.array([sub_df[(sub_df['version'] == v) & (sub_df[sweep_col] == x)]['throughput'].mean()
                                         for x in sweep_values])
            ax_runtime.plot(sweep_values, runtime_means, marker=markers[v], color=colors[v],
                            linewidth=1.8, markersize=7, label=v)
            ax_throughput.plot(sweep_values, throughput_means, marker=markers[v], color=colors[v],
                               linewidth=1.8, markersize=7, label=v)

            if sweep_col == 'frames':
                # Ideal: perfect O(N) runtime → linear, constant throughput.
                # Anchor at the per-version best per-frame rate (smallest runtime/frames ratio).
                best_rate = np.min(runtime_means / sweep_arr)
                ideal_runtime = best_rate * sweep_arr
                ideal_throughput = np.full_like(sweep_arr, np.max(throughput_means))
            else:
                # Ideal: block size shouldn't affect total work → flat at best observed.
                ideal_runtime = np.full_like(sweep_arr, np.min(runtime_means))
                ideal_throughput = np.full_like(sweep_arr, np.max(throughput_means))

            ax_runtime.plot(sweep_values, ideal_runtime, linestyle='--', color=colors[v],
                            linewidth=1.2, alpha=0.6, label=f'{v} ideal')
            ax_throughput.plot(sweep_values, ideal_throughput, linestyle='--', color=colors[v],
                               linewidth=1.2, alpha=0.6, label=f'{v} ideal')

        for ax, ylabel, title in (
            (ax_runtime, 'Runtime (s)', f'Time vs {xlabel} ({title_suffix})'),
            (ax_throughput, 'Throughput (fps)', f'Throughput vs {xlabel} ({title_suffix})'),
        ):
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.set_xticks(sweep_values)
            ax.set_xticklabels(sweep_values)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, which='both', alpha=0.3)
            ax.legend()

    # 1. Frames scaling (log-log)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    frame_sizes = sorted(frames_df['frames'].unique())
    loglog_pair(axes[0], axes[1], frames_df, 'frames', frame_sizes, 'Frames', 'BS=512')
    plt.tight_layout()
    plt.savefig("frames.png")
    plt.show()

    # 2. Block size scaling (log-log)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    block_sizes = sorted(bs_df['block_size'].unique())
    loglog_pair(axes[0], axes[1], bs_df, 'block_size', block_sizes, 'Block Size', 'Frames=2000')
    plt.tight_layout()
    plt.savefig("block-size.png")
    plt.show()

create_plots(df)
