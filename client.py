import OSC
import threading
import time

addr=('127.0.0.1',8000)

c = OSC.OSCClient()
c.connect ( addr )

def sendCounts(counts):
    msg = OSC.OSCMessage()
    msg.setAddress("/counts");
    for i in range(len(counts)):
        msg.append(counts[i])
    c.send(msg)


try:
    while 1:
        s = raw_input ('counts: ')
        counts = [int(i) for i in s.split(' ')]
        sendCounts(counts)
except KeyboardInterrupt:
    pass


