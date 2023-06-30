---
layout: single
author_profile: false
---

There are several tests and tools you can use on Ubuntu to help determine the bottleneck in your download speed. 

1. **Network Speed Tests:** Tools like `speedtest-cli` can be used to test your network speed. Install it using pip:
   ```
   pip install speedtest-cli
   ```
   Then, simply run `speedtest-cli` in your terminal. This will give you an indication of your download and upload speeds.

2. **Disk Speed Tests:** Tools like `dd` and `hdparm` can be used to test the speed of your disk. Here is a simple command using `dd` to test your write speed:
   ```
   dd if=/dev/zero of=tempfile bs=1M count=1024 conv=fdatasync,notrunc
   ```
   This will create a file named "tempfile" in your current directory and measure how long it takes to write it. The `bs` parameter is the block size (1M in this case), and `count` is the number of blocks. So this command writes a 1GB file. You can adjust these values to test with different file sizes.
   
   After the test, don't forget to remove the tempfile using:
   ```
   rm tempfile
   ```
   Alternatively, you can use `hdparm` to test your read speed:
   ```
   sudo hdparm -Tt /dev/sda
   ```
   Replace `/dev/sda` with the path to your SSD. This will give you buffered and cached read speeds.
   
3. **CPU and Memory Utilization:** Tools like `top`, `htop`, or `vmstat` can be used to monitor CPU and memory usage. Simply run `top` or `htop` in your terminal to see a live view of resource usage. The `vmstat` command can also be useful for monitoring system performance.

4. **Network Monitoring:** Tools like `nethogs` can be used to monitor network usage by process. Install it using:
   ```
   sudo apt-get install nethogs
   ```
   Then run `sudo nethogs` to see a live view of network usage. This can help you see if other processes are using a lot of bandwidth.

5. **HuggingFace Specifics:** If you're using HuggingFace's `datasets` library, you could use Python's built-in `cProfile` module to profile your script and see where the most time is being spent. This could help identify if the bottleneck is in the download, the disk write, the decompression, or some other part of the process.

Remember, these tests will give you raw numbers, and it's the interpretation of these numbers that will help you identify the bottleneck. For example, if your network speed is very high but your disk write speed is low, the disk might be the bottleneck. On the other hand, if both your network and disk speeds are high, but you're seeing high CPU usage, the bottleneck might be the CPU.
