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
    import os
    import sys

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
    
    list_all_files = os.listdir(dir_splits_select)
    list_images    =  [ s for s in list_all_files if '.jpeg' in s]
    num_images = len(list_images)    

    img_count = 0
    check_perc  = 0
    check_perc_minor  = 0    
    
    for curr_img in list_images:
        
        # load the image, convert it to grayscale, describe it,
        # and classify it
        img_count += 1            
        curr_perc = float(img_count)/float(num_images) * 100.0       

        if curr_perc >= check_perc:
            sys.stdout.write(str(int(check_perc)))
            check_perc = check_perc + 10.0
            sys.stdout.flush()
    
        if curr_perc >= check_perc_minor:
            sys.stdout.write('.')
            check_perc_minor = check_perc_minor + 2.0
            sys.stdout.flush()
            
        imagePath = dir_splits_select + '/' + curr_img
       
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray)        

        prediction = model.predict(hist.reshape(1, -1))[0]
                
         # display the image and the prediction
          
        if prediction == 'white':
            class_num = 1
       
        if prediction == 'big_boulders':
            class_num = 2           
    
        if prediction == 'crusts':
            class_num = 3

        if prediction == 'small_boulders':
            class_num = 4

        if prediction == 'fine_crust':
            class_num = 5
            
        file_resultfile.write('%s,%d\n' %(imagePath, class_num))
        
    file_resultfile.close()         