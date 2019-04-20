import os
import sys

def process_fn(total_path,output_path):
	cmd='build/bin/FaceLandmarkImg -f ' + total_path + ' -out_dir ' + output_path
	print(cmd)
	os.system(cmd)
	

def run_all_openface(input_list,root_directory,output_root):
	with open(input_list,'r') as f:
		lines=f.readlines()
	for line in lines:
		linesplit=line.split('/')
		subpath=linesplit[1]
		total_path=root_directory + '/' + linesplit[1] + '/' + linesplit[2].strip()
		output_path=output_root+ '/' + subpath
		if not os.path.exists(output_path):
			os.makedirs(output_path)
		cmd='build/bin/FaceLandmarkImg -f ' + total_path + ' -out_dir ' + output_path
		print(cmd)
		os.system(cmd)
if __name__=="__main__":
	run_all_openface(sys.argv[1],sys.argv[2],sys.argv[3])
