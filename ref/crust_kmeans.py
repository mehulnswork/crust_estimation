

def read_statsfile(path_statsfile):

	file_statsfile = open(path_statsfile,'r')
	line = file_statsfile.readline()

	r = []
	g = []
	b = []
	h = []
	s = []
	v = []
	mz = []
	vz = []


	for line in file_statsfile.readlines():

		d = line.split(',')
		r.append(int(d[0]))
		g.append(int(d[1]))
		b.append(int(d[2]))
		h.append(int(d[3]))
		s.append(int(d[4]))
		v.append(int(d[5]))
		mz.append(float(d[6]))
		vz.append(float(d[7]))

	file_statsfile.close()

	print('Number of lines read:' + str(len(r)))
	return(r,g,b,h,s,v,mz,vz)

def param_kmeans(param_matrix, num_clusters):

	from sklearn.cluster import KMeans
	kmeans_res = KMeans(n_clusters = num_clusters, random_state=0).fit(param_matrix)
	
	return kmeans_res.labels_

def write_imagename_class(kmeans_res, path_namesfile, path_resultlist):

	file_namesfile = open(path_namesfile,'r')
	line = file_namesfile.readline()
	
	file_resultfile = open(path_resultlist,'w')
	file_resultfile.write('image, class\n')

	count = 0
	for line in file_namesfile.readlines():

		d = line.split(',')
		res_list = kmeans_res.labels_
		file_resultfile.write('%s,%d\n' %(d[0], res_list[count]))
		count = count + 1

	file_namesfile.close()
	file_resultfile.close()

	print('Number of lines read:' + str(count))


	return None	




