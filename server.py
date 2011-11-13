import OSC
import threading
import time

addr=('127.0.0.1',8000)

def onVoteCounts(*msg):
    print "onVoteCounts", msg
    
s = OSC.OSCServer(addr)
s.addDefaultHandlers()

def countsHandler(addr,tags,stuff,source):
    print '---'
    print 'new msg from %s' % OSC.getUrlStr(source)
    print 'with addr: %s' % addr
    print 'typetags %s' % tags
    print 'data %s' %stuff
    print '---'

s.addMsgHandler("/counts", countsHandler)
s.addMsgHandler("/metacentroids", countsHandler)

st = threading.Thread ( target = s.serve_forever )
st.start()

try:
    while 1:
        time.sleep(5)
except KeyboardInterrupt:
    s.close()
    st.join()
    print 'done'

