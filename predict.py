from utils import *
from configparser import ConfigParser


parser = ConfigParser()
parser.read('config_file.ini')


def z_scores_normalization(img):

	img = (img - np.mean(img)) / np.std(img)
	return img



def load_and_predict_raw_image(hold_out_images,model_name,model_fold, normalization_function = None):

	model = load_model(os.path.join(models_directory,model_fold,model_name))
	
	for filename in hold_out_images:
		filename=filename.strip('\n')

		save_folder=os.path.join(segmentation_directory,os.path.basename(filename),model_fold,model_name)
		print('filename: ',save_folder)		
		flair_nii_gz = load_nii(os.path.join(origin_directory,filename ,'pre','FLAIR.nii.gz'))
		header_info = flair_nii_gz.header
		flair = np.array(flair_nii_gz.get_fdata())
		t1 = load_nii(os.path.join( origin_directory,filename ,'pre', 'T1.nii.gz')).get_data()

		original_shape=t1.shape
	

		if normalization_function != None:

			flair = normalization_function(flair)
			t1 = normalization_function(t1)

		save_flair=flair.copy()	
		save_t1=t1.copy()

		if (t1.shape[2]%16) != 0:
			
			depth_with_padding_z=(math.ceil(t1.shape[2]/16)*16)
			flair = add_padding_z(flair,depth_with_padding_z)
			t1 = add_padding_z(t1,depth_with_padding_z)
			assert t1.shape[2] == depth_with_padding_z
		else:
			depth_with_padding_z = None

		if (t1.shape[1]%16) != 0:
			
			depth_with_padding_y=(math.ceil(t1.shape[1]/16)*16)
			flair = add_padding_y(flair,depth_with_padding_y)
			t1 = add_padding_y(t1,depth_with_padding_y)
			assert t1.shape[1] == depth_with_padding_y

		else:
			depth_with_padding_y = None

		if (t1.shape[0]%16) != 0:
			
			depth_with_padding_x=(math.ceil(t1.shape[0]/16)*16)
			flair = add_padding_x(flair,depth_with_padding_x)
			t1 = add_padding_x(t1,depth_with_padding_x)
			assert t1.shape[0] == depth_with_padding_x

		else:
			depth_with_padding_x = None

	
		# ===== we mix all MRI modalities
		new_shape_all_modalities_joined = (2,t1.shape[0],t1.shape[1],t1.shape[2])
		all_modalities_joined = np.empty(new_shape_all_modalities_joined)
		all_modalities_joined[0]= t1
		all_modalities_joined[1]= flair 


		all_modalities_joined = np.swapaxes(np.swapaxes(np.swapaxes(all_modalities_joined,0,1),1,2),2,3)


		
		all_modalities_list = [all_modalities_joined]

		all_modalities_list = np.array(all_modalities_list, dtype=np.float32)

		y_pred = model.predict(all_modalities_list, batch_size = 1)
	
		if depth_with_padding_x != None:
			y_pred=y_pred[:,np.floor_divide((depth_with_padding_x-original_shape[0]),2):-math.ceil((depth_with_padding_x-original_shape[0])/2),:,:,:]


		if depth_with_padding_y != None:
			y_pred=y_pred[:,:,np.floor_divide((depth_with_padding_y-original_shape[1]),2):-math.ceil((depth_with_padding_y-original_shape[1])/2),:,:]

		if depth_with_padding_z != None:
			y_pred=y_pred[:,:,:,np.floor_divide((depth_with_padding_z-original_shape[2]),2):-math.ceil((depth_with_padding_z-original_shape[2])/2),:]

	
		output_wmh = y_pred.astype(float)[0]
		ensure_dir(save_folder)
		nib.save(nib.Nifti1Image(output_wmh,None,header_info), "{}/wmh_prediction.nii.gz".format(save_folder))
		nib.save(nib.Nifti1Image((output_wmh>0.5).astype(float),None,header_info), "{}/wmh_mask.nii.gz".format(save_folder))			
			

if __name__ == "__main__":


	origin_directory=parser['DEFAULT'].get('image_source_dir')
	segmentation_directory=parser['DEFAULT'].get('segmentation_directory')
	models_directory=parser['DEFAULT'].get('models_directory')
	model_folds=parser['ENSEMBLE'].get('pretrained_models_folds').split(",")
	n_models= parser['ENSEMBLE'].getint('n_models')

	text_file=parser['DEFAULT'].get('hold_out_data')
	hold_out_file = open(text_file, "r")
	hold_out_images = hold_out_file.read().split(' ')
	
	for model_fold in model_folds:
		for i in range(n_models):
		    		
			model_name='model_{}'.format(i)		
			load_and_predict_raw_image(hold_out_images,model_name,model_fold,normalization_function = z_scores_normalization)
	    	    
	
