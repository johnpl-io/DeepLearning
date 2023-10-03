2023-10-02 16:34:34.392340: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
/usr/lib/python3.11/site-packages/h5py/__init__.py:36: UserWarning: h5py is running against HDF5 1.14.2 when it was built against 1.14.1, this may cause problems
  _warn(("h5py is running against HDF5 {0} when it was built against {1}, "
2023-10-02 16:34:35.800506: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:995] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2023-10-02 16:34:35.817367: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:995] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2023-10-02 16:34:35.817574: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:995] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2023-10-02 16:34:35.819214: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:995] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2023-10-02 16:34:35.819375: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:995] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2023-10-02 16:34:35.819513: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:995] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2023-10-02 16:34:35.881114: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:995] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2023-10-02 16:34:35.881315: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:995] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2023-10-02 16:34:35.881461: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:995] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2023-10-02 16:34:35.881578: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 3340 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3070 Ti, pci bus id: 0000:2b:00.0, compute capability: 8.6
2023-10-02 16:34:35.881980: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
2023-10-02 16:34:35.930013: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:432] Loaded cuDNN version 8902
  0%|          | 0/4000 [00:00<?, ?it/s]2023-10-02 16:35:39.077933: I tensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:606] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
2023-10-02 16:35:39.165394: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 902.00MiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
  0%|          | 1/4000 [00:02<2:44:35,  2.47s/it]  0%|          | 2/4000 [00:02<1:19:07,  1.19s/it]  0%|          | 3/4000 [00:03<51:33,  1.29it/s]    0%|          | 4/4000 [00:03<38:35,  1.73it/s]  0%|          | 5/4000 [00:03<31:27,  2.12it/s]  0%|          | 6/4000 [00:03<27:06,  2.46it/s]  0%|          | 7/4000 [00:04<24:21,  2.73it/s]  0%|          | 8/4000 [00:04<22:32,  2.95it/s]  0%|          | 9/4000 [00:04<21:19,  3.12it/s]Step 9; Cost => 2.2177, acc => 0:   0%|          | 9/4000 [00:05<21:19,  3.12it/s]Step 9; Cost => 2.2177, acc => 0:   0%|          | 9/4000 [00:05<21:19,  3.12it/s]Step 9; Cost => 2.2177, acc => 0:   0%|          | 10/4000 [00:05<20:29,  3.25it/s]Step 9; Cost => 2.2177, acc => 0:   0%|          | 11/4000 [00:05<19:54,  3.34it/s]Step 9; Cost => 2.2177, acc => 0:   0%|          | 12/4000 [00:05<19:30,  3.41it/s]Step 9; Cost => 2.2177, acc => 0:   0%|          | 13/4000 [00:05<19:14,  3.45it/s]Step 9; Cost => 2.2177, acc => 0:   0%|          | 14/4000 [00:06<19:02,  3.49it/s]Step 9; Cost => 2.2177, acc => 0:   0%|          | 15/4000 [00:06<18:54,  3.51it/s]Step 9; Cost => 2.2177, acc => 0:   0%|          | 16/4000 [00:06<18:48,  3.53it/s]Step 9; Cost => 2.2177, acc => 0:   0%|          | 17/4000 [00:06<18:44,  3.54it/s]Step 9; Cost => 2.2177, acc => 0:   0%|          | 18/4000 [00:07<18:42,  3.55it/s]Step 9; Cost => 2.2177, acc => 0:   0%|          | 19/4000 [00:07<18:41,  3.55it/s]Step 19; Cost => 2.0480, acc => 0:   0%|          | 19/4000 [00:07<18:41,  3.55it/s]Step 19; Cost => 2.0480, acc => 0:   0%|          | 19/4000 [00:07<18:41,  3.55it/s]Step 19; Cost => 2.0480, acc => 0:   0%|          | 20/4000 [00:07<18:41,  3.55it/s]Step 19; Cost => 2.0480, acc => 0:   1%|          | 21/4000 [00:08<18:47,  3.53it/s]Step 19; Cost => 2.0480, acc => 0:   1%|          | 22/4000 [00:08<18:47,  3.53it/s]Step 19; Cost => 2.0480, acc => 0:   1%|          | 23/4000 [00:08<18:48,  3.52it/s]Step 19; Cost => 2.0480, acc => 0:   1%|          | 24/4000 [00:08<18:48,  3.52it/s]Step 19; Cost => 2.0480, acc => 0:   1%|          | 25/4000 [00:09<18:47,  3.52it/s]Step 19; Cost => 2.0480, acc => 0:   1%|          | 26/4000 [00:09<18:48,  3.52it/s]Step 19; Cost => 2.0480, acc => 0:   1%|          | 27/4000 [00:09<18:48,  3.52it/s]Step 19; Cost => 2.0480, acc => 0:   1%|          | 28/4000 [00:10<18:47,  3.52it/s]Step 19; Cost => 2.0480, acc => 0:   1%|          | 29/4000 [00:10<18:46,  3.52it/s]Step 29; Cost => 1.9618, acc => 0:   1%|          | 29/4000 [00:10<18:46,  3.52it/s]Step 29; Cost => 1.9618, acc => 0:   1%|          | 29/4000 [00:10<18:46,  3.52it/s]Step 29; Cost => 1.9618, acc => 0:   1%|          | 30/4000 [00:10<19:06,  3.46it/s]Step 29; Cost => 1.9618, acc => 0:   1%|          | 31/4000 [00:10<19:15,  3.44it/s]Step 29; Cost => 1.9618, acc => 0:   1%|          | 32/4000 [00:11<19:32,  3.38it/s]Step 29; Cost => 1.9618, acc => 0:   1%|          | 33/4000 [00:11<19:59,  3.31it/s]Step 29; Cost => 1.9618, acc => 0:   1%|          | 34/4000 [00:11<20:13,  3.27it/s]Step 29; Cost => 1.9618, acc => 0:   1%|          | 35/4000 [00:12<20:02,  3.30it/s]Step 29; Cost => 1.9618, acc => 0:   1%|          | 36/4000 [00:12<20:08,  3.28it/s]Step 29; Cost => 1.9618, acc => 0:   1%|          | 37/4000 [00:12<20:12,  3.27it/s]Step 29; Cost => 1.9618, acc => 0:   1%|          | 38/4000 [00:13<20:04,  3.29it/s]Step 29; Cost => 1.9618, acc => 0:   1%|          | 39/4000 [00:13<20:02,  3.29it/s]Step 39; Cost => 1.8869, acc => 0:   1%|          | 39/4000 [00:13<20:02,  3.29it/s]Step 39; Cost => 1.8869, acc => 0:   1%|          | 39/4000 [00:13<20:02,  3.29it/s]Step 39; Cost => 1.8869, acc => 0:   1%|          | 40/4000 [00:13<20:30,  3.22it/s]Step 39; Cost => 1.8869, acc => 0:   1%|          | 41/4000 [00:14<20:33,  3.21it/s]Step 39; Cost => 1.8869, acc => 0:   1%|          | 42/4000 [00:14<20:34,  3.21it/s]Step 39; Cost => 1.8869, acc => 0:   1%|          | 43/4000 [00:14<20:38,  3.20it/s]Step 39; Cost => 1.8869, acc => 0:   1%|          | 44/4000 [00:15<20:39,  3.19it/s]Step 39; Cost => 1.8869, acc => 0:   1%|          | 45/4000 [00:15<20:30,  3.21it/s]Step 39; Cost => 1.8869, acc => 0:   1%|          | 46/4000 [00:15<20:37,  3.19it/s]Step 39; Cost => 1.8869, acc => 0:   1%|          | 47/4000 [00:15<20:46,  3.17it/s]Step 39; Cost => 1.8869, acc => 0:   1%|          | 48/4000 [00:16<20:50,  3.16it/s]Step 39; Cost => 1.8869, acc => 0:   1%|          | 49/4000 [00:16<20:43,  3.18it/s]Step 49; Cost => 1.7826, acc => 0:   1%|          | 49/4000 [00:16<20:43,  3.18it/s]Step 49; Cost => 1.7826, acc => 0:   1%|          | 49/4000 [00:16<20:43,  3.18it/s]Step 49; Cost => 1.7826, acc => 0:   1%|▏         | 50/4000 [00:16<20:46,  3.17it/s]Step 49; Cost => 1.7826, acc => 0:   1%|▏         | 51/4000 [00:17<20:43,  3.17it/s]Step 49; Cost => 1.7826, acc => 0:   1%|▏         | 52/4000 [00:17<21:07,  3.12it/s]Step 49; Cost => 1.7826, acc => 0:   1%|▏         | 53/4000 [00:17<21:02,  3.13it/s]Step 49; Cost => 1.7826, acc => 0:   1%|▏         | 54/4000 [00:18<20:57,  3.14it/s]Step 49; Cost => 1.7826, acc => 0:   1%|▏         | 55/4000 [00:18<21:02,  3.13it/s]Step 49; Cost => 1.7826, acc => 0:   1%|▏         | 56/4000 [00:18<21:06,  3.11it/s]Step 49; Cost => 1.7826, acc => 0:   1%|▏         | 57/4000 [00:19<20:40,  3.18it/s]Step 49; Cost => 1.7826, acc => 0:   1%|▏         | 58/4000 [00:19<20:04,  3.27it/s]Step 49; Cost => 1.7826, acc => 0:   1%|▏         | 59/4000 [00:19<19:42,  3.33it/s]Step 59; Cost => 1.7680, acc => 0:   1%|▏         | 59/4000 [00:20<19:42,  3.33it/s]Step 59; Cost => 1.7680, acc => 0:   1%|▏         | 59/4000 [00:20<19:42,  3.33it/s]Step 59; Cost => 1.7680, acc => 0:   2%|▏         | 60/4000 [00:20<20:01,  3.28it/s]Step 59; Cost => 1.7680, acc => 0:   2%|▏         | 61/4000 [00:20<20:17,  3.23it/s]Step 59; Cost => 1.7680, acc => 0:   2%|▏         | 62/4000 [00:20<20:25,  3.21it/s]Step 59; Cost => 1.7680, acc => 0:   2%|▏         | 63/4000 [00:20<20:34,  3.19it/s]Step 59; Cost => 1.7680, acc => 0:   2%|▏         | 64/4000 [00:21<20:34,  3.19it/s]Step 59; Cost => 1.7680, acc => 0:   2%|▏         | 65/4000 [00:21<20:37,  3.18it/s]Step 59; Cost => 1.7680, acc => 0:   2%|▏         | 66/4000 [00:21<20:47,  3.15it/s]Step 59; Cost => 1.7680, acc => 0:   2%|▏         | 67/4000 [00:22<20:56,  3.13it/s]Step 59; Cost => 1.7680, acc => 0:   2%|▏         | 68/4000 [00:22<20:36,  3.18it/s]Step 59; Cost => 1.7680, acc => 0:   2%|▏         | 69/4000 [00:22<20:21,  3.22it/s]Step 69; Cost => 1.7700, acc => 0:   2%|▏         | 69/4000 [00:23<20:21,  3.22it/s]Step 69; Cost => 1.7700, acc => 0:   2%|▏         | 69/4000 [00:23<20:21,  3.22it/s]Step 69; Cost => 1.7700, acc => 0:   2%|▏         | 70/4000 [00:23<19:50,  3.30it/s]Step 69; Cost => 1.7700, acc => 0:   2%|▏         | 71/4000 [00:23<19:33,  3.35it/s]Step 69; Cost => 1.7700, acc => 0:   2%|▏         | 72/4000 [00:23<19:19,  3.39it/s]Step 69; Cost => 1.7700, acc => 0:   2%|▏         | 73/4000 [00:24<19:29,  3.36it/s]Step 69; Cost => 1.7700, acc => 0:   2%|▏         | 74/4000 [00:24<19:28,  3.36it/s]Step 69; Cost => 1.7700, acc => 0:   2%|▏         | 75/4000 [00:24<19:13,  3.40it/s]Step 69; Cost => 1.7700, acc => 0:   2%|▏         | 76/4000 [00:24<19:08,  3.42it/s]Step 69; Cost => 1.7700, acc => 0:   2%|▏         | 77/4000 [00:25<19:18,  3.39it/s]Step 69; Cost => 1.7700, acc => 0:   2%|▏         | 78/4000 [00:25<19:10,  3.41it/s]Step 69; Cost => 1.7700, acc => 0:   2%|▏         | 79/4000 [00:25<19:05,  3.42it/s]Step 79; Cost => 1.7639, acc => 0:   2%|▏         | 79/4000 [00:26<19:05,  3.42it/s]Step 79; Cost => 1.7639, acc => 0:   2%|▏         | 79/4000 [00:26<19:05,  3.42it/s]Step 79; Cost => 1.7639, acc => 0:   2%|▏         | 80/4000 [00:26<18:58,  3.44it/s]2023-10-02 16:36:14.402881: W tensorflow/tsl/framework/bfc_allocator.cc:485] Allocator (GPU_0_bfc) ran out of memory trying to allocate 585.94MiB (rounded to 614400000)requested by op _EagerConst
If the cause is memory fragmentation maybe the environment variable 'TF_GPU_ALLOCATOR=cuda_malloc_async' will improve the situation. 
Current allocation summary follows.
Current allocation summary follows.
2023-10-02 16:36:14.402957: I tensorflow/tsl/framework/bfc_allocator.cc:1039] BFCAllocator dump for GPU_0_bfc
2023-10-02 16:36:14.402973: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (256): 	Total Chunks: 200, Chunks in use: 197. 50.0KiB allocated for chunks. 49.2KiB in use in bin. 3.9KiB client-requested in use in bin.
2023-10-02 16:36:14.402993: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (512): 	Total Chunks: 39, Chunks in use: 36. 21.0KiB allocated for chunks. 19.2KiB in use in bin. 18.0KiB client-requested in use in bin.
2023-10-02 16:36:14.403012: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (1024): 	Total Chunks: 17, Chunks in use: 14. 19.8KiB allocated for chunks. 15.5KiB in use in bin. 14.0KiB client-requested in use in bin.
2023-10-02 16:36:14.403030: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (2048): 	Total Chunks: 55, Chunks in use: 49. 123.0KiB allocated for chunks. 109.0KiB in use in bin. 98.0KiB client-requested in use in bin.
2023-10-02 16:36:14.403048: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (4096): 	Total Chunks: 2, Chunks in use: 2. 13.5KiB allocated for chunks. 13.5KiB in use in bin. 13.5KiB client-requested in use in bin.
2023-10-02 16:36:14.403068: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (8192): 	Total Chunks: 3, Chunks in use: 3. 26.0KiB allocated for chunks. 26.0KiB in use in bin. 23.5KiB client-requested in use in bin.
2023-10-02 16:36:14.403086: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (16384): 	Total Chunks: 5, Chunks in use: 4. 108.5KiB allocated for chunks. 90.5KiB in use in bin. 80.0KiB client-requested in use in bin.
2023-10-02 16:36:14.403104: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (32768): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2023-10-02 16:36:14.403122: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (65536): 	Total Chunks: 3, Chunks in use: 0. 238.2KiB allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2023-10-02 16:36:14.403140: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (131072): 	Total Chunks: 1, Chunks in use: 1. 160.0KiB allocated for chunks. 160.0KiB in use in bin. 160.0KiB client-requested in use in bin.
2023-10-02 16:36:14.403157: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (262144): 	Total Chunks: 5, Chunks in use: 4. 1.41MiB allocated for chunks. 1.12MiB in use in bin. 1.12MiB client-requested in use in bin.
2023-10-02 16:36:14.403175: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (524288): 	Total Chunks: 17, Chunks in use: 13. 10.25MiB allocated for chunks. 8.06MiB in use in bin. 7.12MiB client-requested in use in bin.
2023-10-02 16:36:14.403193: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (1048576): 	Total Chunks: 9, Chunks in use: 6. 10.63MiB allocated for chunks. 7.01MiB in use in bin. 6.79MiB client-requested in use in bin.
2023-10-02 16:36:14.403210: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (2097152): 	Total Chunks: 2, Chunks in use: 1. 6.97MiB allocated for chunks. 3.00MiB in use in bin. 3.00MiB client-requested in use in bin.
2023-10-02 16:36:14.403228: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (4194304): 	Total Chunks: 10, Chunks in use: 8. 46.50MiB allocated for chunks. 37.50MiB in use in bin. 36.00MiB client-requested in use in bin.
2023-10-02 16:36:14.403257: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (8388608): 	Total Chunks: 5, Chunks in use: 4. 45.00MiB allocated for chunks. 36.00MiB in use in bin. 36.00MiB client-requested in use in bin.
2023-10-02 16:36:14.403277: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (16777216): 	Total Chunks: 1, Chunks in use: 0. 18.00MiB allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2023-10-02 16:36:14.403294: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (33554432): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2023-10-02 16:36:14.403311: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (67108864): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2023-10-02 16:36:14.403329: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (134217728): 	Total Chunks: 2, Chunks in use: 0. 382.52MiB allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2023-10-02 16:36:14.403348: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (268435456): 	Total Chunks: 7, Chunks in use: 1. 2.75GiB allocated for chunks. 468.75MiB in use in bin. 468.75MiB client-requested in use in bin.
2023-10-02 16:36:14.403365: I tensorflow/tsl/framework/bfc_allocator.cc:1062] Bin for 585.94MiB was 256.00MiB, Chunk State: 
2023-10-02 16:36:14.403382: I tensorflow/tsl/framework/bfc_allocator.cc:1068]   Size: 274.42MiB | Requested Size: 4.50MiB | in_use: 0 | bin_num: 20, prev:   Size: 4.50MiB | Requested Size: 4.50MiB | in_use: 1 | bin_num: -1, next:   Size: 256B | Requested Size: 4B | in_use: 1 | bin_num: -1
2023-10-02 16:36:14.403411: I tensorflow/tsl/framework/bfc_allocator.cc:1068]   Size: 277.87MiB | Requested Size: 98.00MiB | in_use: 0 | bin_num: 20, prev:   Size: 160.0KiB | Requested Size: 160.0KiB | in_use: 1 | bin_num: -1, next:   Size: 256B | Requested Size: 1B | in_use: 1 | bin_num: -1
2023-10-02 16:36:14.403438: I tensorflow/tsl/framework/bfc_allocator.cc:1068]   Size: 399.50MiB | Requested Size: 59.26MiB | in_use: 0 | bin_num: 20, prev:   Size: 1.14MiB | Requested Size: 1.14MiB | in_use: 1 | bin_num: -1, next:   Size: 3.00MiB | Requested Size: 3.00MiB | in_use: 1 | bin_num: -1
2023-10-02 16:36:14.403466: I tensorflow/tsl/framework/bfc_allocator.cc:1068]   Size: 425.25MiB | Requested Size: 56.25MiB | in_use: 0 | bin_num: 20, prev:   Size: 468.75MiB | Requested Size: 468.75MiB | in_use: 1 | bin_num: -1, next:   Size: 1.14MiB | Requested Size: 1.14MiB | in_use: 1 | bin_num: -1
2023-10-02 16:36:14.403493: I tensorflow/tsl/framework/bfc_allocator.cc:1068]   Size: 440.00MiB | Requested Size: 392.53MiB | in_use: 0 | bin_num: 20, prev:   Size: 1.14MiB | Requested Size: 1.14MiB | in_use: 1 | bin_num: -1, next:   Size: 4.50MiB | Requested Size: 4.50MiB | in_use: 1 | bin_num: -1
2023-10-02 16:36:14.403521: I tensorflow/tsl/framework/bfc_allocator.cc:1068]   Size: 532.88MiB | Requested Size: 278.75MiB | in_use: 0 | bin_num: 20, prev:   Size: 9.00MiB | Requested Size: 9.00MiB | in_use: 1 | bin_num: -1, next:   Size: 1.14MiB | Requested Size: 1.14MiB | in_use: 1 | bin_num: -1
2023-10-02 16:36:14.403545: I tensorflow/tsl/framework/bfc_allocator.cc:1075] Next region of size 3502964736
2023-10-02 16:36:14.403559: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1c8a000000 of size 1280 next 1
2023-10-02 16:36:14.403571: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1c8a000500 of size 655360 next 2
2023-10-02 16:36:14.403582: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1c8a0a0500 of size 163840 next 3
2023-10-02 16:36:14.403593: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1c8a0c8500 of size 291372032 next 40011
2023-10-02 16:36:14.403612: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1c9b6a8100 of size 256 next 40013
2023-10-02 16:36:14.403623: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1c9b6a8200 of size 491520000 next 40014
2023-10-02 16:36:14.403646: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1cb8b68200 of size 445907712 next 4
2023-10-02 16:36:14.403658: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1cd34a8500 of size 1200128 next 5
2023-10-02 16:36:14.403671: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1cd35cd500 of size 418906112 next 76
2023-10-02 16:36:14.403684: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1cec54d500 of size 3145728 next 140
2023-10-02 16:36:14.403697: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1cec84d500 of size 6291456 next 406
2023-10-02 16:36:14.403710: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1cece4d500 of size 18874368 next 332
2023-10-02 16:36:14.403721: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1cee04d500 of size 4718592 next 398
2023-10-02 16:36:14.403730: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1cee4cd500 of size 4718592 next 446
2023-10-02 16:36:14.403738: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1cee94d500 of size 4718592 next 151
2023-10-02 16:36:14.403747: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1ceedcd500 of size 9437184 next 401
2023-10-02 16:36:14.403756: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1cef6cd500 of size 9437184 next 231
2023-10-02 16:36:14.403764: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1ceffcd500 of size 9437184 next 131
2023-10-02 16:36:14.403773: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1cf08cd500 of size 170917888 next 413
2023-10-02 16:36:14.403781: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1cfabcd500 of size 9437184 next 248
2023-10-02 16:36:14.403792: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1cfb4cd500 of size 558759936 next 6
2023-10-02 16:36:14.403802: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d1c9ad500 of size 1200128 next 7
2023-10-02 16:36:14.403812: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d1cad2500 of size 461373440 next 124
2023-10-02 16:36:14.403823: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d382d2500 of size 4718592 next 323
2023-10-02 16:36:14.403833: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d38752500 of size 4718592 next 362
2023-10-02 16:36:14.403843: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d38bd2500 of size 4718592 next 448
2023-10-02 16:36:14.403854: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d39052500 of size 4718592 next 417
2023-10-02 16:36:14.403864: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d394d2500 of size 287752192 next 8
2023-10-02 16:36:14.403874: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a73e500 of size 256 next 9
2023-10-02 16:36:14.403885: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a73e600 of size 256 next 10
2023-10-02 16:36:14.403895: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a73e700 of size 256 next 11
2023-10-02 16:36:14.403905: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a73e800 of size 256 next 12
2023-10-02 16:36:14.403915: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a73e900 of size 256 next 15
2023-10-02 16:36:14.403926: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a73ea00 of size 256 next 13
2023-10-02 16:36:14.403936: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a73eb00 of size 256 next 14
2023-10-02 16:36:14.403946: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a73ec00 of size 256 next 18
2023-10-02 16:36:14.403956: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a73ed00 of size 256 next 16
2023-10-02 16:36:14.403966: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a73ee00 of size 256 next 20
2023-10-02 16:36:14.403977: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a73ef00 of size 256 next 19
2023-10-02 16:36:14.403987: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a73f000 of size 256 next 21
2023-10-02 16:36:14.404000: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a73f100 of size 256 next 23
2023-10-02 16:36:14.404012: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a73f200 of size 512 next 24
2023-10-02 16:36:14.404022: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a73f400 of size 512 next 25
2023-10-02 16:36:14.404032: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a73f600 of size 512 next 30
2023-10-02 16:36:14.404042: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a73f800 of size 256 next 28
2023-10-02 16:36:14.404053: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a73f900 of size 512 next 32
2023-10-02 16:36:14.404063: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a73fb00 of size 512 next 35
2023-10-02 16:36:14.404073: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a73fd00 of size 512 next 33
2023-10-02 16:36:14.404083: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a73ff00 of size 512 next 34
2023-10-02 16:36:14.404093: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a740100 of size 512 next 36
2023-10-02 16:36:14.404103: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a740300 of size 512 next 39
2023-10-02 16:36:14.404114: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a740500 of size 1024 next 38
2023-10-02 16:36:14.404124: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a740900 of size 1024 next 42
2023-10-02 16:36:14.404135: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a740d00 of size 1024 next 41
2023-10-02 16:36:14.404145: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a741100 of size 256 next 43
2023-10-02 16:36:14.404155: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a741200 of size 2048 next 47
2023-10-02 16:36:14.404166: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a741a00 of size 2560 next 17
2023-10-02 16:36:14.404176: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a742400 of size 6912 next 22
2023-10-02 16:36:14.404186: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a743f00 of size 2048 next 45
2023-10-02 16:36:14.404196: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a744700 of size 2048 next 48
2023-10-02 16:36:14.404206: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a744f00 of size 2048 next 51
2023-10-02 16:36:14.404217: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a745700 of size 2048 next 54
2023-10-02 16:36:14.404227: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a745f00 of size 2048 next 52
2023-10-02 16:36:14.404237: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a746700 of size 2048 next 56
2023-10-02 16:36:14.404247: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a746f00 of size 2048 next 58
2023-10-02 16:36:14.404257: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a747700 of size 2048 next 55
2023-10-02 16:36:14.404267: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a747f00 of size 2048 next 57
2023-10-02 16:36:14.404278: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a748700 of size 256 next 61
2023-10-02 16:36:14.404288: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a748800 of size 2048 next 59
2023-10-02 16:36:14.404298: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a749000 of size 20480 next 62
2023-10-02 16:36:14.404308: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74e000 of size 256 next 63
2023-10-02 16:36:14.404317: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74e100 of size 256 next 73
2023-10-02 16:36:14.404326: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74e200 of size 256 next 170
2023-10-02 16:36:14.404335: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74e300 of size 256 next 173
2023-10-02 16:36:14.404347: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74e400 of size 256 next 70
2023-10-02 16:36:14.404356: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74e500 of size 256 next 286
2023-10-02 16:36:14.404365: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74e600 of size 256 next 90
2023-10-02 16:36:14.404374: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74e700 of size 256 next 65
2023-10-02 16:36:14.404382: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74e800 of size 256 next 316
2023-10-02 16:36:14.404391: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74e900 of size 256 next 300
2023-10-02 16:36:14.404400: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74ea00 of size 256 next 297
2023-10-02 16:36:14.404408: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74eb00 of size 256 next 107
2023-10-02 16:36:14.404416: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74ec00 of size 256 next 341
2023-10-02 16:36:14.404425: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74ed00 of size 256 next 91
2023-10-02 16:36:14.404433: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74ee00 of size 256 next 119
2023-10-02 16:36:14.404444: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74ef00 of size 256 next 317
2023-10-02 16:36:14.404455: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74f000 of size 256 next 182
2023-10-02 16:36:14.404465: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74f100 of size 256 next 249
2023-10-02 16:36:14.404476: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74f200 of size 256 next 168
2023-10-02 16:36:14.404486: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74f300 of size 256 next 94
2023-10-02 16:36:14.404497: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74f400 of size 256 next 262
2023-10-02 16:36:14.404507: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74f500 of size 256 next 164
2023-10-02 16:36:14.404517: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74f600 of size 256 next 236
2023-10-02 16:36:14.404527: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74f700 of size 256 next 304
2023-10-02 16:36:14.404537: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74f800 of size 256 next 246
2023-10-02 16:36:14.404547: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74f900 of size 256 next 189
2023-10-02 16:36:14.404557: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74fa00 of size 256 next 83
2023-10-02 16:36:14.404568: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74fb00 of size 256 next 161
2023-10-02 16:36:14.404579: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74fc00 of size 256 next 299
2023-10-02 16:36:14.404589: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74fd00 of size 256 next 318
2023-10-02 16:36:14.404599: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74fe00 of size 256 next 287
2023-10-02 16:36:14.404622: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a74ff00 of size 256 next 280
2023-10-02 16:36:14.404633: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a750000 of size 256 next 319
2023-10-02 16:36:14.404642: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a750100 of size 256 next 77
2023-10-02 16:36:14.404650: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a750200 of size 256 next 220
2023-10-02 16:36:14.404659: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a750300 of size 256 next 82
2023-10-02 16:36:14.404668: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a750400 of size 256 next 201
2023-10-02 16:36:14.404681: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a750500 of size 256 next 268
2023-10-02 16:36:14.404692: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a750600 of size 256 next 276
2023-10-02 16:36:14.404703: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a750700 of size 256 next 376
2023-10-02 16:36:14.404713: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a750800 of size 256 next 359
2023-10-02 16:36:14.404723: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a750900 of size 256 next 103
2023-10-02 16:36:14.404734: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a750a00 of size 256 next 308
2023-10-02 16:36:14.404744: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a750b00 of size 256 next 79
2023-10-02 16:36:14.404754: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a750c00 of size 256 next 158
2023-10-02 16:36:14.404764: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a750d00 of size 256 next 288
2023-10-02 16:36:14.404775: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a750e00 of size 256 next 105
2023-10-02 16:36:14.404785: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a750f00 of size 256 next 163
2023-10-02 16:36:14.404795: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a751000 of size 256 next 130
2023-10-02 16:36:14.404805: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a751100 of size 256 next 223
2023-10-02 16:36:14.404815: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a751200 of size 256 next 295
2023-10-02 16:36:14.404825: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a751300 of size 256 next 146
2023-10-02 16:36:14.404835: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a751400 of size 256 next 235
2023-10-02 16:36:14.404846: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a751500 of size 256 next 183
2023-10-02 16:36:14.404856: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a751600 of size 256 next 134
2023-10-02 16:36:14.404866: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a751700 of size 256 next 355
2023-10-02 16:36:14.404877: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a751800 of size 256 next 251
2023-10-02 16:36:14.404887: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a751900 of size 256 next 365
2023-10-02 16:36:14.404897: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a751a00 of size 256 next 154
2023-10-02 16:36:14.404907: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a751b00 of size 256 next 67
2023-10-02 16:36:14.404917: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a751c00 of size 256 next 212
2023-10-02 16:36:14.404928: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a751d00 of size 256 next 368
2023-10-02 16:36:14.404938: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a751e00 of size 256 next 279
2023-10-02 16:36:14.404948: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a751f00 of size 256 next 363
2023-10-02 16:36:14.404958: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a752000 of size 256 next 116
2023-10-02 16:36:14.404968: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a752100 of size 256 next 218
2023-10-02 16:36:14.404978: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a752200 of size 256 next 132
2023-10-02 16:36:14.404989: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a752300 of size 256 next 245
2023-10-02 16:36:14.404999: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a752400 of size 256 next 258
2023-10-02 16:36:14.405009: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a752500 of size 256 next 414
2023-10-02 16:36:14.405019: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a752600 of size 256 next 204
2023-10-02 16:36:14.405035: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a752700 of size 256 next 429
2023-10-02 16:36:14.405046: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a752800 of size 256 next 75
2023-10-02 16:36:14.405057: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4a752900 of size 256 next 430
2023-10-02 16:36:14.405067: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a752a00 of size 512 next 121
2023-10-02 16:36:14.405077: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a752c00 of size 256 next 432
2023-10-02 16:36:14.405087: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a752d00 of size 256 next 174
2023-10-02 16:36:14.405097: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4a752e00 of size 512 next 162
2023-10-02 16:36:14.405107: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a753000 of size 768 next 255
2023-10-02 16:36:14.405118: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a753300 of size 256 next 114
2023-10-02 16:36:14.405128: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a753400 of size 256 next 129
2023-10-02 16:36:14.405138: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a753500 of size 256 next 337
2023-10-02 16:36:14.405148: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a753600 of size 256 next 87
2023-10-02 16:36:14.405158: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a753700 of size 256 next 282
2023-10-02 16:36:14.405168: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a753800 of size 256 next 274
2023-10-02 16:36:14.405178: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a753900 of size 256 next 278
2023-10-02 16:36:14.405189: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a753a00 of size 256 next 115
2023-10-02 16:36:14.405199: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a753b00 of size 256 next 237
2023-10-02 16:36:14.405209: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a753c00 of size 256 next 302
2023-10-02 16:36:14.405219: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a753d00 of size 256 next 144
2023-10-02 16:36:14.405230: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a753e00 of size 256 next 415
2023-10-02 16:36:14.405240: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a753f00 of size 256 next 102
2023-10-02 16:36:14.405250: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a754000 of size 256 next 181
2023-10-02 16:36:14.405260: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a754100 of size 256 next 89
2023-10-02 16:36:14.405270: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a754200 of size 256 next 208
2023-10-02 16:36:14.405281: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a754300 of size 256 next 407
2023-10-02 16:36:14.405291: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a754400 of size 256 next 98
2023-10-02 16:36:14.405301: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a754500 of size 256 next 385
2023-10-02 16:36:14.405311: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4a754600 of size 512 next 66
2023-10-02 16:36:14.405321: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a754800 of size 256 next 225
2023-10-02 16:36:14.405332: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a754900 of size 256 next 85
2023-10-02 16:36:14.405342: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a754a00 of size 256 next 234
2023-10-02 16:36:14.405352: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a754b00 of size 256 next 81
2023-10-02 16:36:14.405362: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a754c00 of size 256 next 145
2023-10-02 16:36:14.405370: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a754d00 of size 256 next 93
2023-10-02 16:36:14.405383: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a754e00 of size 256 next 277
2023-10-02 16:36:14.405391: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a754f00 of size 256 next 250
2023-10-02 16:36:14.405400: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a755000 of size 256 next 117
2023-10-02 16:36:14.405409: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a755100 of size 256 next 388
2023-10-02 16:36:14.405417: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a755200 of size 256 next 198
2023-10-02 16:36:14.405426: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a755300 of size 256 next 244
2023-10-02 16:36:14.405434: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a755400 of size 256 next 127
2023-10-02 16:36:14.405443: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a755500 of size 256 next 283
2023-10-02 16:36:14.405451: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a755600 of size 256 next 123
2023-10-02 16:36:14.405460: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a755700 of size 256 next 197
2023-10-02 16:36:14.405468: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a755800 of size 256 next 179
2023-10-02 16:36:14.405476: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a755900 of size 256 next 109
2023-10-02 16:36:14.405485: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a755a00 of size 256 next 354
2023-10-02 16:36:14.405493: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a755b00 of size 256 next 185
2023-10-02 16:36:14.405502: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a755c00 of size 256 next 99
2023-10-02 16:36:14.405511: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a755d00 of size 256 next 97
2023-10-02 16:36:14.405519: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a755e00 of size 256 next 292
2023-10-02 16:36:14.405527: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a755f00 of size 256 next 175
2023-10-02 16:36:14.405536: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a756000 of size 256 next 95
2023-10-02 16:36:14.405546: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a756100 of size 256 next 186
2023-10-02 16:36:14.405557: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a756200 of size 256 next 386
2023-10-02 16:36:14.405567: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a756300 of size 256 next 333
2023-10-02 16:36:14.405577: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a756400 of size 256 next 312
2023-10-02 16:36:14.405587: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a756500 of size 256 next 226
2023-10-02 16:36:14.405597: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a756600 of size 512 next 243
2023-10-02 16:36:14.405607: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a756800 of size 768 next 135
2023-10-02 16:36:14.405618: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a756b00 of size 256 next 80
2023-10-02 16:36:14.405629: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a756c00 of size 256 next 141
2023-10-02 16:36:14.405639: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a756d00 of size 256 next 137
2023-10-02 16:36:14.405649: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a756e00 of size 256 next 387
2023-10-02 16:36:14.405659: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a756f00 of size 256 next 326
2023-10-02 16:36:14.405670: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a757000 of size 256 next 147
2023-10-02 16:36:14.405680: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a757100 of size 256 next 348
2023-10-02 16:36:14.405690: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a757200 of size 768 next 442
2023-10-02 16:36:14.405704: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a757500 of size 256 next 242
2023-10-02 16:36:14.405713: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a757600 of size 256 next 424
2023-10-02 16:36:14.405722: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a757700 of size 3328 next 353
2023-10-02 16:36:14.405733: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a758400 of size 256 next 265
2023-10-02 16:36:14.405743: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a758500 of size 256 next 335
2023-10-02 16:36:14.405754: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a758600 of size 256 next 273
2023-10-02 16:36:14.405764: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a758700 of size 256 next 206
2023-10-02 16:36:14.405774: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a758800 of size 256 next 241
2023-10-02 16:36:14.405785: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a758900 of size 256 next 125
2023-10-02 16:36:14.405795: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a758a00 of size 256 next 72
2023-10-02 16:36:14.405805: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a758b00 of size 256 next 128
2023-10-02 16:36:14.405815: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4a758c00 of size 2816 next 324
2023-10-02 16:36:14.405826: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a759700 of size 256 next 291
2023-10-02 16:36:14.405836: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a759800 of size 256 next 371
2023-10-02 16:36:14.405846: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a759900 of size 256 next 165
2023-10-02 16:36:14.405856: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a759a00 of size 256 next 169
2023-10-02 16:36:14.405866: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a759b00 of size 256 next 331
2023-10-02 16:36:14.405876: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a759c00 of size 256 next 233
2023-10-02 16:36:14.405887: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a759d00 of size 256 next 338
2023-10-02 16:36:14.405897: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a759e00 of size 2048 next 266
2023-10-02 16:36:14.405907: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75a600 of size 2304 next 298
2023-10-02 16:36:14.405917: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75af00 of size 256 next 443
2023-10-02 16:36:14.405928: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75b000 of size 256 next 339
2023-10-02 16:36:14.405938: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75b100 of size 512 next 260
2023-10-02 16:36:14.405948: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75b300 of size 512 next 374
2023-10-02 16:36:14.405958: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75b500 of size 512 next 360
2023-10-02 16:36:14.405968: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4a75b700 of size 256 next 310
2023-10-02 16:36:14.405978: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75b800 of size 256 next 423
2023-10-02 16:36:14.405988: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75b900 of size 512 next 408
2023-10-02 16:36:14.405998: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75bb00 of size 512 next 88
2023-10-02 16:36:14.406009: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75bd00 of size 3328 next 358
2023-10-02 16:36:14.406018: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75ca00 of size 768 next 271
2023-10-02 16:36:14.406026: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75cd00 of size 256 next 440
2023-10-02 16:36:14.406035: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75ce00 of size 256 next 184
2023-10-02 16:36:14.406048: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75cf00 of size 256 next 187
2023-10-02 16:36:14.406059: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75d000 of size 256 next 188
2023-10-02 16:36:14.406070: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75d100 of size 256 next 190
2023-10-02 16:36:14.406080: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75d200 of size 256 next 191
2023-10-02 16:36:14.406089: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75d300 of size 256 next 194
2023-10-02 16:36:14.406098: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75d400 of size 256 next 192
2023-10-02 16:36:14.406106: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75d500 of size 256 next 196
2023-10-02 16:36:14.406115: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75d600 of size 256 next 193
2023-10-02 16:36:14.406123: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75d700 of size 256 next 195
2023-10-02 16:36:14.406132: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75d800 of size 256 next 68
2023-10-02 16:36:14.406143: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75d900 of size 2048 next 96
2023-10-02 16:36:14.406153: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75e100 of size 2560 next 238
2023-10-02 16:36:14.406163: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75eb00 of size 1024 next 425
2023-10-02 16:36:14.406173: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75ef00 of size 512 next 252
2023-10-02 16:36:14.406184: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75f100 of size 1024 next 428
2023-10-02 16:36:14.406194: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75f500 of size 768 next 399
2023-10-02 16:36:14.406204: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75f800 of size 256 next 336
2023-10-02 16:36:14.406214: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75f900 of size 1024 next 84
2023-10-02 16:36:14.406224: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a75fd00 of size 1024 next 221
2023-10-02 16:36:14.406234: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4a760100 of size 1536 next 416
2023-10-02 16:36:14.406244: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a760700 of size 256 next 110
2023-10-02 16:36:14.406254: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a760800 of size 256 next 139
2023-10-02 16:36:14.406264: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a760900 of size 256 next 216
2023-10-02 16:36:14.406274: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a760a00 of size 512 next 439
2023-10-02 16:36:14.406285: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a760c00 of size 1024 next 100
2023-10-02 16:36:14.406295: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a761000 of size 1536 next 256
2023-10-02 16:36:14.406305: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a761600 of size 3840 next 361
2023-10-02 16:36:14.406314: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a762500 of size 256 next 149
2023-10-02 16:36:14.406323: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a762600 of size 256 next 454
2023-10-02 16:36:14.406331: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4a762700 of size 256 next 176
2023-10-02 16:36:14.406340: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a762800 of size 256 next 344
2023-10-02 16:36:14.406348: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a762900 of size 512 next 157
2023-10-02 16:36:14.406357: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4a762b00 of size 768 next 384
2023-10-02 16:36:14.406365: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a762e00 of size 512 next 422
2023-10-02 16:36:14.406378: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4a763000 of size 1024 next 420
2023-10-02 16:36:14.406389: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a763400 of size 1536 next 257
2023-10-02 16:36:14.406400: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a763a00 of size 512 next 382
2023-10-02 16:36:14.406410: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a763c00 of size 512 next 156
2023-10-02 16:36:14.406420: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a763e00 of size 1280 next 270
2023-10-02 16:36:14.406430: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4a764300 of size 2048 next 375
2023-10-02 16:36:14.406441: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a764b00 of size 8192 next 284
2023-10-02 16:36:14.406450: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a766b00 of size 2048 next 410
2023-10-02 16:36:14.406459: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a767300 of size 512 next 366
2023-10-02 16:36:14.406467: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a767500 of size 512 next 294
2023-10-02 16:36:14.406476: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a767700 of size 512 next 113
2023-10-02 16:36:14.406485: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a767900 of size 512 next 285
2023-10-02 16:36:14.406495: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a767b00 of size 512 next 126
2023-10-02 16:36:14.406506: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a767d00 of size 256 next 152
2023-10-02 16:36:14.406516: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a767e00 of size 256 next 431
2023-10-02 16:36:14.406527: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a767f00 of size 256 next 340
2023-10-02 16:36:14.406537: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a768000 of size 1024 next 327
2023-10-02 16:36:14.406547: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4a768400 of size 1792 next 101
2023-10-02 16:36:14.406557: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a768b00 of size 2048 next 78
2023-10-02 16:36:14.406567: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a769300 of size 2048 next 445
2023-10-02 16:36:14.406577: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4a769b00 of size 66304 next 373
2023-10-02 16:36:14.406587: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a779e00 of size 2048 next 172
2023-10-02 16:36:14.406598: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a77a600 of size 2048 next 400
2023-10-02 16:36:14.406608: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a77ae00 of size 2048 next 138
2023-10-02 16:36:14.406618: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a77b600 of size 2048 next 180
2023-10-02 16:36:14.406628: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a77be00 of size 10240 next 329
2023-10-02 16:36:14.406638: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a77e600 of size 2048 next 378
2023-10-02 16:36:14.406649: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a77ee00 of size 8192 next 381
2023-10-02 16:36:14.406659: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a780e00 of size 20480 next 325
2023-10-02 16:36:14.406670: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4a785e00 of size 2048 next 392
2023-10-02 16:36:14.406680: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a786600 of size 2048 next 437
2023-10-02 16:36:14.406690: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a786e00 of size 2816 next 217
2023-10-02 16:36:14.406700: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a787900 of size 6912 next 389
2023-10-02 16:36:14.406714: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a789400 of size 20992 next 281
2023-10-02 16:36:14.406725: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a78e600 of size 256 next 228
2023-10-02 16:36:14.406736: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a78e700 of size 3072 next 352
2023-10-02 16:36:14.406746: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a78f300 of size 3072 next 205
2023-10-02 16:36:14.406756: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a78ff00 of size 256 next 209
2023-10-02 16:36:14.406766: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a790000 of size 256 next 253
2023-10-02 16:36:14.406776: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a790100 of size 256 next 240
2023-10-02 16:36:14.406786: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a790200 of size 256 next 142
2023-10-02 16:36:14.406797: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a790300 of size 256 next 86
2023-10-02 16:36:14.406807: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a790400 of size 512 next 419
2023-10-02 16:36:14.406817: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a790600 of size 512 next 347
2023-10-02 16:36:14.406827: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4a790800 of size 2048 next 342
2023-10-02 16:36:14.406838: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a791000 of size 3072 next 405
2023-10-02 16:36:14.406848: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a791c00 of size 2048 next 314
2023-10-02 16:36:14.406858: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a792400 of size 2048 next 293
2023-10-02 16:36:14.406868: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a792c00 of size 1024 next 441
2023-10-02 16:36:14.406878: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a793000 of size 512 next 322
2023-10-02 16:36:14.406889: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a793200 of size 512 next 409
2023-10-02 16:36:14.406898: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4a793400 of size 92416 next 412
2023-10-02 16:36:14.406907: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a7a9d00 of size 2048 next 118
2023-10-02 16:36:14.406915: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a7aa500 of size 2048 next 111
2023-10-02 16:36:14.406924: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4a7aad00 of size 3328 next 239
2023-10-02 16:36:14.406935: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a7aba00 of size 2048 next 159
2023-10-02 16:36:14.406945: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a7ac200 of size 2048 next 307
2023-10-02 16:36:14.406956: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a7aca00 of size 2048 next 92
2023-10-02 16:36:14.406966: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a7ad200 of size 2048 next 106
2023-10-02 16:36:14.406976: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a7ada00 of size 2816 next 370
2023-10-02 16:36:14.406986: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a7ae500 of size 2048 next 213
2023-10-02 16:36:14.406996: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a7aed00 of size 2048 next 320
2023-10-02 16:36:14.407007: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a7af500 of size 2048 next 69
2023-10-02 16:36:14.407017: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a7afd00 of size 2048 next 447
2023-10-02 16:36:14.407027: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a7b0500 of size 2048 next 455
2023-10-02 16:36:14.407037: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a7b0d00 of size 3072 next 247
2023-10-02 16:36:14.407047: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a7b1900 of size 256 next 290
2023-10-02 16:36:14.407061: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4a7b1a00 of size 2048 next 232
2023-10-02 16:36:14.407072: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a7b2200 of size 2048 next 104
2023-10-02 16:36:14.407083: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a7b2a00 of size 2048 next 426
2023-10-02 16:36:14.407093: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4a7b3200 of size 18432 next 449
2023-10-02 16:36:14.407103: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a7b7a00 of size 30720 next 411
2023-10-02 16:36:14.407113: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4a7bf200 of size 85248 next 26
2023-10-02 16:36:14.407124: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a7d3f00 of size 294912 next 27
2023-10-02 16:36:14.407134: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a81bf00 of size 294912 next 269
2023-10-02 16:36:14.407144: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4a863f00 of size 294912 next 29
2023-10-02 16:36:14.407154: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a8abf00 of size 589824 next 31
2023-10-02 16:36:14.407164: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a93bf00 of size 589824 next 37
2023-10-02 16:36:14.407174: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4a9cbf00 of size 524288 next 50
2023-10-02 16:36:14.407185: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4aa4bf00 of size 589824 next 391
2023-10-02 16:36:14.407195: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4aadbf00 of size 589824 next 403
2023-10-02 16:36:14.407205: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4ab6bf00 of size 655360 next 40
2023-10-02 16:36:14.407215: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4ac0bf00 of size 1179648 next 44
2023-10-02 16:36:14.407226: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4ad2bf00 of size 1179648 next 321
2023-10-02 16:36:14.407236: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4ae4bf00 of size 1179648 next 436
2023-10-02 16:36:14.407247: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4af6bf00 of size 1409024 next 444
2023-10-02 16:36:14.407257: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4b0c3f00 of size 589824 next 350
2023-10-02 16:36:14.407268: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4b153f00 of size 524288 next 261
2023-10-02 16:36:14.407278: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4b1d3f00 of size 589824 next 330
2023-10-02 16:36:14.407288: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4b263f00 of size 294912 next 379
2023-10-02 16:36:14.407299: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4b2abf00 of size 950272 next 296
2023-10-02 16:36:14.407309: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4b393f00 of size 589824 next 390
2023-10-02 16:36:14.407320: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4b423f00 of size 970752 next 393
2023-10-02 16:36:14.407331: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4b510f00 of size 524288 next 166
2023-10-02 16:36:14.407341: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4b590f00 of size 634880 next 49
2023-10-02 16:36:14.407351: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4b62bf00 of size 4718592 next 46
2023-10-02 16:36:14.407362: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4baabf00 of size 4718592 next 53
2023-10-02 16:36:14.407372: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4bf2bf00 of size 1179648 next 367
2023-10-02 16:36:14.407380: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4c04bf00 of size 1179648 next 155
2023-10-02 16:36:14.407389: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4c16bf00 of size 1441792 next 402
2023-10-02 16:36:14.407403: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4c2cbf00 of size 589824 next 219
2023-10-02 16:36:14.407414: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4c35bf00 of size 589824 next 364
2023-10-02 16:36:14.407422: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4c3ebf00 of size 294912 next 421
2023-10-02 16:36:14.407431: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4c433f00 of size 4161536 next 60
2023-10-02 16:36:14.407440: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f1d4c82bf00 of size 9437184 next 64
2023-10-02 16:36:14.407449: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f1d4d12bf00 of size 230179072 next 18446744073709551615
2023-10-02 16:36:14.407464: I tensorflow/tsl/framework/bfc_allocator.cc:1100]      Summary of in-use Chunks by size: 
2023-10-02 16:36:14.407477: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 197 Chunks of size 256 totalling 49.2KiB
2023-10-02 16:36:14.407487: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 31 Chunks of size 512 totalling 15.5KiB
2023-10-02 16:36:14.407497: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 5 Chunks of size 768 totalling 3.8KiB
2023-10-02 16:36:14.407506: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 10 Chunks of size 1024 totalling 10.0KiB
2023-10-02 16:36:14.407515: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 1280 totalling 2.5KiB
2023-10-02 16:36:14.407524: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 1536 totalling 3.0KiB
2023-10-02 16:36:14.407534: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 37 Chunks of size 2048 totalling 74.0KiB
2023-10-02 16:36:14.407545: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 2304 totalling 2.2KiB
2023-10-02 16:36:14.407554: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 2560 totalling 5.0KiB
2023-10-02 16:36:14.407564: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 2816 totalling 5.5KiB
2023-10-02 16:36:14.407573: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 4 Chunks of size 3072 totalling 12.0KiB
2023-10-02 16:36:14.407584: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 3328 totalling 6.5KiB
2023-10-02 16:36:14.407595: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 3840 totalling 3.8KiB
2023-10-02 16:36:14.407604: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 6912 totalling 13.5KiB
2023-10-02 16:36:14.407614: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 8192 totalling 16.0KiB
2023-10-02 16:36:14.407623: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 10240 totalling 10.0KiB
2023-10-02 16:36:14.407634: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 20480 totalling 40.0KiB
2023-10-02 16:36:14.407645: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 20992 totalling 20.5KiB
2023-10-02 16:36:14.407656: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 30720 totalling 30.0KiB
2023-10-02 16:36:14.407665: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 163840 totalling 160.0KiB
2023-10-02 16:36:14.407674: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 4 Chunks of size 294912 totalling 1.12MiB
2023-10-02 16:36:14.407684: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 524288 totalling 1.00MiB
2023-10-02 16:36:14.407693: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 6 Chunks of size 589824 totalling 3.38MiB
2023-10-02 16:36:14.407702: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 634880 totalling 620.0KiB
2023-10-02 16:36:14.407713: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 655360 totalling 1.25MiB
2023-10-02 16:36:14.407723: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 950272 totalling 928.0KiB
2023-10-02 16:36:14.407734: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 970752 totalling 948.0KiB
2023-10-02 16:36:14.407749: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 3 Chunks of size 1179648 totalling 3.38MiB
2023-10-02 16:36:14.407760: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 1200128 totalling 2.29MiB
2023-10-02 16:36:14.407770: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 1409024 totalling 1.34MiB
2023-10-02 16:36:14.407781: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 3145728 totalling 3.00MiB
2023-10-02 16:36:14.407792: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 7 Chunks of size 4718592 totalling 31.50MiB
2023-10-02 16:36:14.407802: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 6291456 totalling 6.00MiB
2023-10-02 16:36:14.407812: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 4 Chunks of size 9437184 totalling 36.00MiB
2023-10-02 16:36:14.407822: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 491520000 totalling 468.75MiB
2023-10-02 16:36:14.407831: I tensorflow/tsl/framework/bfc_allocator.cc:1107] Sum Total of in-use chunks: 561.92MiB
2023-10-02 16:36:14.407841: I tensorflow/tsl/framework/bfc_allocator.cc:1109] Total bytes in pool: 3502964736 memory_limit_: 3502964736 available bytes: 0 curr_region_allocation_bytes_: 7005929472
2023-10-02 16:36:14.407857: I tensorflow/tsl/framework/bfc_allocator.cc:1114] Stats: 
Limit:                      3502964736
InUse:                       589212672
MaxInUse:                   3288551168
NumAllocs:                      469363
MaxAllocSize:               1228800000
Reserved:                            0
PeakReserved:                        0
LargestFreeBlock:                    0

2023-10-02 16:36:14.407887: W tensorflow/tsl/framework/bfc_allocator.cc:497] *_______***************____________*___________***___**_______________*____________*________**______
Step 79; Cost => 1.7639, acc => 0:   2%|▏         | 80/4000 [00:36<29:33,  2.21it/s]
Traceback (most recent call last):
  File "/usr/lib/python3.11/site-packages/tensorflow/python/ops/array_ops.py", line 5136, in gather
    return params.sparse_read(indices, name=name)
           ^^^^^^^^^^^^^^^^^^
AttributeError: 'numpy.ndarray' object has no attribute 'sparse_read'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/john/Documents/DeepLearning/hw4/cifar10.py", line 123, in <module>
    train_images_batch = tf.gather(features, batch_indices)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/site-packages/tensorflow/python/util/traceback_utils.py", line 150, in error_handler
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/site-packages/tensorflow/python/util/dispatch.py", line 1176, in op_dispatch_handler
    return dispatch_target(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/site-packages/tensorflow/python/ops/array_ops.py", line 5149, in gather_v2
    return gather(
           ^^^^^^^
  File "/usr/lib/python3.11/site-packages/tensorflow/python/util/traceback_utils.py", line 150, in error_handler
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/site-packages/tensorflow/python/util/dispatch.py", line 1176, in op_dispatch_handler
    return dispatch_target(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/site-packages/tensorflow/python/util/deprecation.py", line 576, in new_func
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/site-packages/tensorflow/python/ops/array_ops.py", line 5138, in gather
    return gen_array_ops.gather_v2(params, indices, axis, name=name)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/site-packages/tensorflow/python/ops/gen_array_ops.py", line 3965, in gather_v2
    _result = pywrap_tfe.TFE_Py_FastPathExecute(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/site-packages/tensorflow/python/framework/errors_impl.py", line 462, in __init__
    def __init__(self, node_def, op, message, *args):
  
KeyboardInterrupt
