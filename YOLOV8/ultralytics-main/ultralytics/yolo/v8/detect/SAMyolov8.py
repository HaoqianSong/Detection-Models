from ultralytics import SAM
import cv2 as cv

model = SAM('sam_b.pt')
model.info()
result = model.predict('D:/DetectionAlgorithm/DataSet/05_01_0459.jpg')

model.info()
result = model.predict('D:/DetectionAlgorithm/DataSet/05_01_0459.jpg', save=False)
frame = result[0].orig_img
reimg = result[0].plot()
cv.waitKey(0)
cv.destroyAllWindows()
