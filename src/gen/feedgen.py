#!/usr/bin/env python3
# feedgen.py -- simple UDP tick generator (CSV lines)
# Usage: python3 feedgen.py [host] [port] [rate_hz]
# print("HELLO")
import socket, time, random, sys

HOST = sys.argv[1] if len(sys.argv) > 1 else '127.0.0.1'
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9000
RATE_HZ = float(sys.argv[3]) if len(sys.argv) > 3 else 2000.0

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
addr = (HOST, PORT)

seq = 0
price = 100.0

interval = 1.0 / RATE_HZ
print(f"Sending ticks to {HOST}:{PORT} at {RATE_HZ} hz (interval {interval:.6f}s)")

try:
    while True:
        ts = time.time()
        price += random.gauss(-0.5, 0.5)
        size = random.randint(1, 1000)
        line = f"{seq},{ts:.9f},{price:.6f},{size}\n"
        sock.sendto(line.encode('utf-8'), addr)
        seq += 1
        time.sleep(interval)

except KeyboardInterrupt:
    print("\nTerminated by user.")
    sock.close()
    sys.exit(0)