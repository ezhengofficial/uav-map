import airsim, time

c = airsim.MultirotorClient()
c.confirmConnection()
c.enableApiControl(True)
c.armDisarm(True)

print("Taking off...")
c.takeoffAsync().join()
c.moveToZAsync(-5, 2).join()   # hover about 5 m above ground
time.sleep(3)
c.landAsync().join()

c.armDisarm(False)
c.enableApiControl(False)
print("Visual test complete.")