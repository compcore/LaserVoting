import cv
import cv2
import numpy
import OSC
import os, sys, time

threshold = 24
minSpotSize = 1

counts = []
countEstimate = []
countAlpha = 0.3

centroids = []
metacentroids = []
trackedMetacentroids = []
trackingAlpha = 0.25
oscClient = None
frame = 0
testImages = []
addr = ( ('127.0.0.1'),8000 )
#addr = ( ('134.154.11.12'),8000 )
background = None
grayBG = None
grabBackground = True
hotSpots = []
autoScalers = []

class HotSpot:
    def __init__(self, ccam, rcam ):
        # center in normalized camera coords
        self.c = numpy.array([ccam[0],ccam[1]])
        self.r = rcam

    def getCenter ( self ):
        return self.c
    
    def inside ( self, spot ):
        # spot in normalized camera coords
        d = abs(spot-self.c)
        if numpy.linalg.norm(d) < self.r:
            return 1
        else:
            return 0

    def draw ( self, img ):
        sz = (img.shape[1], img.shape[0])
        c = self.c * sz
        cv2.circle ( img, (int(c[0]),int(c[1])), int(self.r * sz[0]), (255,255,255) )

    def toEye ( self, pt ):
        # convert to coords with hotspot center at origin
        eyePt = numpy.array ( [ pt[0], pt[1] ] )
        eyePt = ( eyePt - self.c ) / self.r
        return eyePt

class SquareHotSpot:
    def __init__ ( self, ccam, w, h ):
        self.c = numpy.array([ccam[0],ccam[1]])
        self.w = w
        self.h = h

    def getCenter ( self ):
        return self.c

    def inside ( self, spot ):
        innit = False
        if spot[0] > self.c[0] - self.w/2 \
                and spot[0] < self.c[0] + self.w/2 \
                and spot[1] > self.c[1] - self.h/2 \
                and spot[1] < self.c[1] + self.h/2:
            innit = True
        return innit

    def draw ( self, img ) :
        sz=(img.shape[1],img.shape[0])
        c = self.c * sz
        w = self.w * sz[0]
        h = self.h * sz[1]
        pt1 = c - (w/2, h/2)
        pt2 = c + (w/2, h/2)
        cv2.rectangle ( img, (int(pt1[0]),int(pt1[1])), 
                       (int(pt2[0]),int(pt2[1])), (255,255,255) )

    def toEye ( self, pt ):
        eyePt = numpy.array ( [ pt[0], pt[1] ] )
        eyePt = ( eyePt - self.c )
        eyePt /= (self.w/2, self.h/2)
        return eyePt

class AutoScaler :
    def __init__ ( self ):
        self._min = numpy.array([10.0,10.0])
        self._max = numpy.array([-10.0,-10.0])

    def update ( self, npos ):
        for i in range(2):
            self._min[i] = min ( npos[i], self._min[i] )
            self._max[i] = max ( npos[i], self._max[i] )

    def rescale ( self, npos ):
        result = numpy.array([1.,1.])
        for i in range(2):
            s = abs(self._max[i] - self._min[i]);
#            print 'min, s ', self._min[i], s
            result[i] = (npos[i] - self._min[i]) / s
        return result

def captureTest():
    import cv2
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print """can't open capture"""
        return
    win=cv2.namedWindow('pic',cv2.CV_GUI_NORMAL)
    while cv2.waitKey(1):
        ok, img = cap.read()
        if not ok:
            print """can't retrieve()"""
        else:
            cv2.imshow('pic',img)
    cap.release()
    cv2.destroyWindow('pic')


def loadTestImages ():
    global testImages
    for f in range(21):
        path = os.path.join ( 'testImages', '%010d.ppm' % f )
        print path
        testImages.append (cv2.LoadImage ( path ))
        if testImages[f] == None:
            print 'error loading test image %s' % path
    

def locateTest( capture, window ):
    centroids = []
    ok,img = capture.read()
    imgSize = cv2.GetSize(img)
    thresh = cv2.CreateImage ( imgSize, 8, 1 )
    cv2.CvtColor ( img, thresh, int(cv2.CV_RGB2GRAY) )
    cv2.Threshold ( thresh, thresh, threshold, 255, cv2.THRESH_BINARY )
    seq = cv2.FindContours ( thresh, cv2.CreateMemStorage() )
    return seq

def locateSpots ( capture, window, threshwindow, hotSpots ):
    global centroids,counts, frame, testImages, metacentroids, background, grayBG
    global grabBackground
    
    centroids = [] # list of lists
    counts=[] # list of int counts
    for i in range(len(hotSpots)):
        counts.append(0)
        centroids.append([])
        
    # get a frame and find the contours
    if capture != None :
        #        for i in range(2): # wtf better at least
        #           img = cv2.QueryFrame(capture)
        ok,frame = capture.read()
    else:
        frame = testImages[frame % len(testImages)]
        frame += 1
        time.sleep(1)

    
    img=frame
    imgSize = (img.shape[1], img.shape[0])
#    thresh = cv.CreateImage ( imgSize, 8, 1 )
    thresh = cv.CreateImage ( imgSize, cv2.IPL_DEPTH_32F, 1 )
    subtracted = img.copy()
#    cv2.CvtColor ( img, thresh, cv2.CV_RGB2GRAY )

    if grabBackground:
        print 'grab back'
#        background = img.copy()
#        grayBG = cv2.cvtColor ( background, cv2.COLOR_RGB2GRAY )
        grayBG = cv2.cvtColor ( img, cv2.COLOR_RGB2GRAY )
        grabBackground = False

    # background subtraction
    #cv.Sub(img,background,subtracted)

#    subtracted = img - background

#    cv2.circle ( subtracted, (20,20), 5, (255,255,255) )
    cv2.circle ( img, (20,20), 5, (255,255,255) )
    
#    gray = cv.CreateImage ( imgSize, 8, 1 )
#    gray = cv.CreateImage ( imgSize, cv2.IPL_DEPTH_32F, 1 )
#    gray = cv2.cvtColor ( subtracted, cv2.COLOR_RGB2GRAY )
    grayimg = cv2.cvtColor ( img, cv2.COLOR_RGB2GRAY )
    gray = grayimg - grayBG
#    gray = gray*gray
    gray[gray<0] = 0
    gray[gray>120] = 0

    print 'gray[0,0]', gray[0,0]
    # threshold
#    ok, thresh = cv2.threshold ( gray, threshold, 255.0, cv2.THRESH_BINARY )
    ok, thresh = cv2.threshold ( gray, threshold, 255.0, cv2.THRESH_BINARY )
    thresholded = thresh.copy()
    cv2.imshow(threshwindow, thresholded)
    
#    storage = cv2.CreateMemStorage()
    
    contours = cv2.findContours ( thresh,  cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE )
        
     # test each contour against the hotspots, accumulate counts

    n=len(contours[0])
    print "spots found:", n
    x = 0.0
    y = 0.0
#    while n>0 and contour != None:
    for i in range(n):
#        print 'contours[i]', contours[0][i]
        moments = cv2.moments ( contours[0][i] )
#        print 'moments', moments
        x = moments['m10']
        y = moments['m01']
        sum = moments['m00']

        if sum > minSpotSize:
            x /= sum
            y /= sum
            ix = int ( x + 0.5 )
            iy = int ( y + 0.5 )
            cv2.line ( img, (ix-5,iy), (ix+5,iy), (0,0,255), 1, cv2.CV_AA )
            cv2.line ( img, (ix,iy-5), (ix,iy+5), (0,0,255), 1, cv2.CV_AA )
            for i in range(len(hotSpots)):
                npos = (x/imgSize[0],y/imgSize[1])
                innit = hotSpots[i].inside(npos)
                counts[i] += innit
                if innit:
                    centroids[i].append(numpy.array(npos))
                    
#                counts[i] += hotSpots[i].inside((x/imgSize[0],y/imgSize[1]))

#        contour = contour.h_next()

    print img.shape
    for spot in hotSpots:
        spot.draw(img)

#    for i in range(len(hotSpots)):
#        print 'count['+str(i)+'] = '+str(counts[i])

    # find the centroids, per hotspot, of the detected centroids
    metacentroids = []
    for i in range(len(hotSpots)):
        metacentroids.append ( numpy.array([0.0,0.0]) )
        for j in range(len(centroids[i])):
            metacentroids[i] += centroids[i][j]
        if len(centroids[i])==0:
            metacentroids[i] = numpy.array([-1.0,-1.0])
        else:
            metacentroids[i] /= len(centroids[i])

    trackMetacentroids()
    
    for i in range(len(metacentroids)):
        c = metacentroids[i] * imgSize
        cv2.circle ( img, (int(c[0]),int(c[1])), 5, (0,255,0) )
        t = trackedMetacentroids[i] * imgSize
        cv2.circle ( img, (int(t[0]),int(t[1])), 10, (0,255,255) )
        
    cv2.imshow ( window, img )
    

def onTrackbar ( x ):
    global threshold
    threshold = x

    
def oscHotSpot (*msg):
    print "oscHotSpot", msg

    
def oscClear (*msg):
    print "oscHotSpot", msg

    
def sendCounts():
    msg = OSC.OSCMessage()
    msg.setAddress("/counts")
    for i in range(len(counts)):
        msg.append(counts[i])
    oscClient.send(msg)

def sendCountEstimates():
    # update estimates
    for i in range(len(counts)):
        countEstimate[i] = countAlpha * counts[i] + (1.0-countAlpha) * countEstimate[i]
    msg = OSC.OSCMessage()
    msg.setAddress("/counts")
    for i in range(len(counts)):
        msg.append(int(countEstimate[i]+0.5)) #rounded
    oscClient.send(msg)
    

# def sendCentroids():
#     msg = OSC.OSCMessage()
#     msg.setAddress("/centroids")
#     for i in range(len(centroids)):
#         msg.append(centroids[i][0])
#         msg.append(centroids[i][1])
#     oscClient.send(msg)

def sendMetaCentroids():
# send normalized metacentroids
    msg = OSC.OSCMessage()
    msg.setAddress("/metacentroids")
    for i in range(len(metacentroids)):
#        npos = (metacentroids[i] - hotSpots[i].c) / hotSpots[i].r
        npos = hotSpots[i].toEye(metacentroids[i]);
        msg.append(npos[0])
        msg.append(npos[1])
    oscClient.send(msg)

def trackMetacentroids():
    global trackedMetacentroids, trackingAlpha, metacentroids
    for i in range(len(metacentroids)):
#        print 'metacentroid['+str(i)+'] ', metacentroids[i]
        c = numpy.array(metacentroids[i])
        t = trackedMetacentroids[i]
#        print 'tracked ',t
        if c[0]==-1.0 and c[1]==-1.0:
            trackedMetacentroids[i] = c
        else:
            if t[0]==-1.0 and t[1]==-1.0:
                trackedMetacentroids[i] = c
            else:
                trackedMetacentroids[i] = trackingAlpha * c + (1.0 - trackingAlpha) * t

def sendTrackedMetaCentroids():
    global hotSpots, trackedMetacentroids, metacentroids, oscClient
    msg = OSC.OSCMessage()
    msg.setAddress("/metacentroids")
    for i in range(len(trackedMetacentroids)):
        if trackedMetacentroids[i][0] == -1. and trackedMetacentroids[i][1] == -1. :
            msg.append(-1.)
            msg.append(-1.)
        else:
            # find position in normalized HotSpot coordinates
#            npos = (trackedMetacentroids[i] - hotSpots[i].c) / hotSpots[i].r
            npos = hotSpots[i].toEye(trackedMetacentroids[i])
            npos += (1.0,1.0)
            npos /= 2.0
            if npos[0] < 0 or npos[0] > 1 or npos[1] < 0 or npos[1] > 1:
                print 'bad npos?', npos
            # update auto scaling
            autoScalers[i].update ( npos );
            # rescale within range seen so far
            npos = autoScalers[i].rescale ( npos );

#            print 'after rescale ', npos
            
            # pack into message
            msg.append(npos[0])
            msg.append(1.0-npos[1])
    oscClient.send(msg)

def installHotSpots ( newHotSpots ) :
    global hotSpots, trackedMetacentroids, autoScalers
    
    hotSpots = newHotSpots
                           
    trackedMetacentroids=[]
    autoScalers=[]
    for i in range (len(hotSpots)):
        trackedMetacentroids.append ( hotSpots[i].getCenter() )
        autoScalers.append ( AutoScaler() )


        
def main():
    global oscClient, trackedMetacentroids, trackingAlpha, grabBackground
    
    oscClient = OSC.OSCClient()
    oscClient.connect( addr )

    if len(sys.argv) == 2:
        if sys.argv[1].isdigit():
            capture = cv2.VideoCapture ( int(sys.argv[1]) )
        else:
            capture = cv2.VideoCapture ( sys.argv[1] )
    else:
        loadTestImages()
        capture = None # use test images
        
    cv2.namedWindow ( 'spots'  )
    cv2.namedWindow ( 'thresh' )
    cv2.createTrackbar ( 'threshold', 'thresh', threshold, 255, onTrackbar)

    pongHotSpots = []
    pongY = 0.6
    pongHotSpots.append(HotSpot((0.15,pongY),0.15))
    pongHotSpots.append(HotSpot((0.4, pongY),0.1))
    pongHotSpots.append(HotSpot((0.6, pongY),0.1))
    pongHotSpots.append(HotSpot((0.85, pongY),0.15))

    votingHotSpots = []
    votingHotSpots.append(SquareHotSpot((0.15,0.5),0.3,0.2))
    votingHotSpots.append(SquareHotSpot((0.4, 0.5), 0.2, 0.2))
    votingHotSpots.append(SquareHotSpot((0.6, 0.5), 0.2, 0.2))
    votingHotSpots.append(SquareHotSpot((0.85, 0.5), 0.3, 0.2))

    drivingHotSpots = []
#    drivingHotSpots.append(HotSpot((0.5,0.4), 0.06))
    drivingHotSpots.append(SquareHotSpot((0.5,0.5), 0.9, 0.15))

    bigHotSpot = []
    bigHotSpot.append(HotSpot((0.5,0.5), 0.4))

#    installHotSpots ( pongHotSpots )
#    installHotSpots ( drivingHotSpots )
    installHotSpots ( pongHotSpots )

    for i in range(4):
        countEstimate.append(0.0)
        
    key = cv2.waitKey(1)
    while key != ord('q'):

        locateSpots ( capture, 'spots', 'thresh', hotSpots )
#        sendCounts ()
        sendCountEstimates ()
        sendTrackedMetaCentroids()

        if key == ord('g'):
            grabBackground = True
            installHotSpots ( hotSpots )

        if key == ord('d'):
            installHotSpots ( drivingHotSpots )

        if key == ord('p'):
            installHotSpots ( pongHotSpots )

        if key == ord('b'):
            installHotSpots ( bigHotSpot )

        if key == ord('v'):
            installHotSpots ( votingHotSpots )

        key = cv2.waitKey(1)


        

if __name__ == '__main__':
    main()
    
