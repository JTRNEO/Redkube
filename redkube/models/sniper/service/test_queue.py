import rediswq
import re
import os

q=rediswq.RedisWQ(name="job", host="redis")
print("Worker with sessionID: " +  q.sessionID())
print("Initial queue state: empty=" + str(q.empty()))
        
while not q.empty():
    item = q.lease(lease_secs=60, block=False)
    if item is not None:
            itemstr = item.decode("utf=8")
            itemstr = re.findall(r"\d+\.?\d*",itemstr)
            slice_name = "{}_{}.png".format(itemstr[0],itemstr[1])
            with open('/sniper/output.txt','a') as fi:
                 fi.write("Working on slice: {}".format(slice_name))
            print("Working on slice :" + slice_name )
            q.complete(item)
    else:
        print("waiting for work")
