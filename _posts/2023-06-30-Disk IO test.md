---
layout: single
author_profile: false
---

### Disk Details
| Disk name                                            | Storage type    | Size (GiB) | Max IOPS | Max throughput (MBps) | Encryption    | Host caching |
|------------------------------------------------------|-----------------|------------|----------|-----------------------|---------------|--------------|
| V100_OsDisk_1_9d9847279b55436b92dba603cd7bd366        | Premium SSD LRS | 1024       | 5000     | 200                   | SSE with PMK  | Read/Write   |

### File system overview
```bash
(base) jingchao@V100:~$ df -h
Filesystem      Size  Used Avail Use% Mounted on
/dev/root       993G  509G  484G  52% /
devtmpfs        221G     0  221G   0% /dev
tmpfs           221G     0  221G   0% /dev/shm
tmpfs            45G  1.5M   45G   1% /run
tmpfs           5.0M     0  5.0M   0% /run/lock
tmpfs           221G     0  221G   0% /sys/fs/cgroup
/dev/loop0       64M   64M     0 100% /snap/core20/1822
/dev/sda15      105M  6.1M   99M   6% /boot/efi
/dev/loop1       64M   64M     0 100% /snap/core20/1950
/dev/loop2       92M   92M     0 100% /snap/lxd/24061
/dev/loop3       54M   54M     0 100% /snap/snapd/19457
/dev/loop4       50M   50M     0 100% /snap/snapd/18357
/dev/sdb1       2.9T  192M  2.9T   1% /mnt
```

There are several tests and tools you can use on Ubuntu to help determine the bottleneck in your download speed. 

1. **Network Speed Tests:** Tools like `speedtest-cli` can be used to test your network speed. Install it using pip:
   ```
   pip install speedtest-cli
   ```
   Then, simply run `speedtest-cli` in your terminal. This will give you an indication of your download and upload speeds.

   ```bash
   (base) jingchao@V100:~$ speedtest-cli --bytes
   Retrieving speedtest.net configuration...
   Testing from Microsoft Corporation (20.225.51.252)...
   Retrieving speedtest.net server list...
   Selecting best server based on ping...
   Hosted by United Cooperative Services – DA3 (Burleson, TX) [589.57 km]: 15.037 ms
   Testing download speed................................................................................
   Download: 218.03 Mbyte/s
   Testing upload speed......................................................................................................
   Upload: 166.72 Mbyte/s
   ```

3. **Disk Speed Tests:** Tools like `dd` and `hdparm` can be used to test the speed of your disk. Here is a simple command using `dd` to test your write speed:
   ```
   dd if=/dev/zero of=tempfile bs=1M count=1024 conv=fdatasync,notrunc
   ```
   This will create a file named "tempfile" in your current directory and measure how long it takes to write it. The `bs` parameter is the block size (1M in this case), and `count` is the number of blocks. So this command writes a 1GB file. You can adjust these values to test with different file sizes.

   `/dev/zero` is a special file in Unix-like operating systems that produces as many null bytes (bytes with a value of zero) as are read from it. 
   
   If you want to write random data instead of null bytes, you can use `/dev/urandom` as the input file. `/dev/urandom` is another special file that produces random bytes when read. Here's how you would modify the `dd` command:
   
   ```bash
   dd if=/dev/urandom of=tempfile bs=1M count=1024 conv=fdatasync,notrunc
   ```
   
   Keep in mind that writing random data will generally be slower than writing null bytes, because generating random data requires some computational effort. As a result, this test might give a lower write speed, but it might also be a more accurate representation of real-world disk write performance. After you've done the test, don't forget to remove the tempfile with `rm tempfile`.
   
   ```bash
   (base) jingchao@V100:~$ dd if=/dev/zero of=tempfile bs=1M count=1024 conv=fdatasync,notrunc
   1024+0 records in
   1024+0 records out
   1073741824 bytes (1.1 GB, 1.0 GiB) copied, 5.38708 s, 199 MB/s
   (base) jingchao@V100:~$ 
   ```
   ```bash
   (base) jingchao@V100:~$ dd if=/dev/urandom of=tempfile bs=1M count=1024 conv=fdatasync,notrunc
   1024+0 records in
   1024+0 records out
   1073741824 bytes (1.1 GB, 1.0 GiB) copied, 8.58883 s, 125 MB/s
   (base) jingchao@V100:~$ 
   ```

   
   After the test, don't forget to remove the tempfile using:
   ```
   rm tempfile
   ```
   Alternatively, you can use `hdparm` to test your read speed:
   ```
   sudo hdparm -Tt /dev/sda
   ```
   Replace `/dev/sda` with the path to your SSD. This will give you buffered and cached read speeds.
   
   ```bash
   (base) jingchao@V100:~$ sudo hdparm -Tt /dev/sda
   
   /dev/sda:
    Timing cached reads:   20018 MB in  1.98 seconds = 10092.45 MB/sec
   SG_IO: bad/missing sense data, sb[]:  70 00 05 00 00 00 00 0a 00 00 00 00 20 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
    Timing buffered disk reads: 746 MB in  3.01 seconds = 247.88 MB/sec
   (base) jingchao@V100:~$ 
   ```
   
4. **CPU and Memory Utilization:** Tools like `top`, `htop`, or `vmstat` can be used to monitor CPU and memory usage. Simply run `top` or `htop` in your terminal to see a live view of resource usage. The `vmstat` command can also be useful for monitoring system performance.

5. **Network Monitoring:** Tools like `nethogs` can be used to monitor network usage by process. Install it using:
   ```
   sudo apt-get install nethogs
   ```
   Then run `sudo nethogs` to see a live view of network usage. This can help you see if other processes are using a lot of bandwidth.

7. **HuggingFace Specifics:** If you're using HuggingFace's `datasets` library, you could use Python's built-in `cProfile` module to profile your script and see where the most time is being spent. This could help identify if the bottleneck is in the download, the disk write, the decompression, or some other part of the process.

Remember, these tests will give you raw numbers, and it's the interpretation of these numbers that will help you identify the bottleneck. For example, if your network speed is very high but your disk write speed is low, the disk might be the bottleneck. On the other hand, if both your network and disk speeds are high, but you're seeing high CPU usage, the bottleneck might be the CPU.
