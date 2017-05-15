# import the necessary packages

 
class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
 
    def describe(self, image, eps=1e-7):
        from skimage import feature
        import numpy as np
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
            self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.numPoints + 3),
            range=(0, self.numPoints + 2))
 
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
 
        # return the histogram of Local Binary Patterns
        return hist
        

def func(dir_training, dir_splits_select, path_resultlist):
    
    print('Classificaton of texture\n')

    from sklearn.svm import LinearSVC
    from imutils import paths
    import cv2

    desc = LocalBinaryPatterns(24, 8)
    data = []
    labels = []

    for imagePath in paths.list_images(dir_training):
        # load the image, convert it to grayscale, and describe it
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray)
     
        # extract the label from the image path, then update the
        # label and data lists
        labels.append(imagePath.split("/")[-2])
        data.append(hist)
     
    # train a Linear SVM on the data
    model = LinearSVC(C=100.0, random_state=42)
    model.fit(data, labels)
    
    # loop over the testing images
    
    file_resultfile = open(path_resultlist, 'w')
    file_resultfile.write('image, class\n')
    
    for imagePath in paths.list_images(dir_splits_select):
        # load the image, convert it to grayscale, describe it,
        # and classify it
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray)
        prediction = model.predict(hist.reshape(1, -1))[0]
                
         # display the image and the prediction
          
        if prediction == 'white':
            class_num = 0
       
        if prediction == 'boulders':
            class_num = 1           
    
        if prediction == 'crusts':
            class_num = 2
        
        file_resultfile.write('%s,%d\n' %(imagePath, class_num))
        
    file_resultfile.close()         