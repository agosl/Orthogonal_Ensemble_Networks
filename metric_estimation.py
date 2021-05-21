from utils import*
from configparser import ConfigParser





def get_ensemble_variance(case_folder,subject_id,mfiles=None,segment=False):
    
    ensemble_mean_prob= get_wmh_probabilities(case_folder,ensemble=True,mfiles=mfiles)
    if segment==True:	
        ensemble_lm=	ensemble_label_map=(ensemble_mean_prob>0.5).astype(float) 
        positive_pixels=(ensemble_lm==1).astype(float)
    variance_estimation=0


    for mfile in mfiles:
    	probability_file = os.path.join(case_folder+str(mfile),'wmh_prediction.nii.gz')
    	prob_image = nib.load(probability_file).get_fdata()[:,:,:,0]
    	variance_image=np.square(prob_image-ensemble_mean_prob)

    	if segment:
    		variance=np.sum(variance_image*positive_pixels)/np.sum(positive_pixels)
    	else:
    		variance=np.mean(variance_image)    	
    	variance_estimation += variance

    return variance_estimation/len(mfiles)





def get_wmh_probabilities(case_folder,ensemble=False,mfiles=None):
    if ensemble:
 	   probability_file = os.path.join(case_folder+str(0),'wmh_prediction.nii.gz')
 	   img_shape = (nib.load(probability_file).get_fdata()).shape
 	   probability_prediction=np.zeros((img_shape[0],img_shape[1],img_shape[2]))
 	   for mfile in mfiles:
    		probability_file = os.path.join(case_folder+str(mfile),'wmh_prediction.nii.gz')
    		prob_image = nib.load(probability_file).get_fdata()
    		probability_prediction += prob_image[:,:,:,0]
 
 	   return np.around(probability_prediction/len(mfiles),4)
    else:
	    probability_file = os.path.join(case_folder,'wmh_prediction.nii.gz')
	    probability_image = nib.load(probability_file)
	    prob_img = probability_image.get_fdata()
	    return prob_img

def dice_coefficient(gt, prediction):

	return 2 * np.sum(gt * prediction)/(np.sum(gt) + np.sum(prediction))  

def load_segmentation(case_folder,subject_id,gt=False):
	if gt== False:
	    prediction_file = "{}/wmh_mask.nii.gz".format(case_folder)
	    prediction_image = nib.load(prediction_file)
	    prediction = prediction_image.get_fdata()
	    print('Loading') 	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
	    return prediction[:,:,:,0]
	else:
	    gt_file = "{}{}/wmh.nii.gz".format(case_folder,subject_id)
	    gt_image = nib.load(gt_file)
	    gt = gt_image.get_fdata()
	    gt=set2to0(gt)
	    return gt

def ensemble_segmentacion(case_folder,M):
	ensemble_probs=get_wmh_probabilities(case_folder,ensemble=True,mfiles=M)
	ensemble_label_map=(ensemble_probs>0.5).astype(float)
	return ensemble_label_map



def brier_score(case_folder,subject_id,ensemble=False,mfiles=None):
 
    gt=load_segmentation(gt_directory,subject_id,gt=True).flatten()
    probabilities=get_wmh_probabilities(case_folder,ensemble=ensemble,mfiles=mfiles).flatten()
    bs = brier_score_loss(gt,probabilities)
    return bs

def brier_plus(case_folder,subject_id,ensemble=False,mfiles=None):
    if ensemble==True:
    	gt=load_segmentation(gt_directory,subject_id,gt=True).flatten()
	
    else:
    	gt=load_segmentation(gt_directory,subject_id,gt=True).flatten()
    foreground_voxels=gt[[gt.astype(float)==1]]
    probabilities=get_wmh_probabilities(case_folder,ensemble=ensemble,mfiles=mfiles).flatten()

    foreground_probabilities=probabilities[[gt.astype(float)==1]]
    if len(foreground_probabilities)>0:
    	bs = brier_score_loss(foreground_voxels,foreground_probabilities)
    else:
    	bs= 'NaN'
    return bs

def save_ensemble_pred(case_folder,model_fold,subject_id,mfiles):
    probs=get_wmh_probabilities(case_folder,ensemble=True,mfiles=mfiles)
    ensure_dir("./ensemble_preds/{}/{}".format(subject_id,model_fold))
    nib.save(nib.Nifti1Image(probs,None), "./ensemble_preds/{}/{}/ensemble_wmh.nii.gz".format(subject_id,model_fold))
    nib.save(nib.Nifti1Image((probs>0.5).astype(float),None), "./ensemble_preds/{}/{}/wmh_mask.nii.gz".format(subject_id,model_fold))



def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
 
def str2net(v):

    if v.lower() in ('resnet', 'resunet', 'ResUnet', 'ResUNet'):
        return 'ResUNet_'
    elif v.lower() in ('Unet', 'unet', 'UNet'):
        return 'UNet_'
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def testing_models(metrics,model_fold,n_models,folder,save_results_fold):

	for metric in metrics:


			if metric=='dice':
			    	header = ["dice"]
			elif metric=='brier_plus':    
				header = ["brier_plus"]
			elif metric=='brier':    
				header = ["brier"]  
  
   
			else:
				raise SystemExit("The metric does not exist")
			model_mean=list()
			model_names= list()

			for model_number in range(n_models):
			    rows = list()
			    subject_ids = list()
			    model_name = 'model_{}'.format(model_number)
			    model_names.append(model_name)
			   
			    for base_case_folder in glob(folder+'/*'):

				    subject_id= os.path.basename(base_case_folder)
				    subject_ids.append(subject_id)
				    case_folder=os.path.join(base_case_folder,model_fold,model_name)

				    if metric=='dice':
				         hard_dice = [dice_coefficient(load_segmentation(gt_directory,subject_id,True), load_segmentation(case_folder,subject_id))]
				         print('Hard_dice_anatomic: ',hard_dice)
				         rows.append(hard_dice)
				    elif metric=='brier_plus':     
				         brier= [brier_plus(case_folder,subject_id) ]   
				         print('Brier Score: ',brier)
				         rows.append(brier)      
				    elif metric=='brier':     
				         brier= [brier_score(case_folder,subject_id) ]   
				         print('Brier Score: ',brier)
				         rows.append(brier) 

			    df = pd.DataFrame.from_records(rows, columns=header, index=subject_ids)
			    ensure_dir(os.path.join('./'+save_results_fold,model_fold))
			    df.to_csv(os.path.join('./'+save_results_fold,model_fold,metric+'_model_{}.csv'.format(model_number)))
			    model_mean.append(df.mean())
			df_means = pd.DataFrame.from_records(model_mean, columns=header, index=model_names)
			df_means.to_csv(os.path.join(save_results_fold,model_fold,'mean_'+metric+'_model.csv'))  
			
        
       


 

def testing_ensemble(Nnet,kcross,metrics,model_fold,n_models,folder,save_results_fold):

	for metric in metrics:

		if metric=='dice':
		    	header = ["dice"]
		elif metric=='brier_plus':    
			header = ["brier_plus"]
		elif metric=='brier':    
			header = ["brier"] 
		elif metric=='variance':    
			header = ["variance"] 
  
		else:
			raise SystemExit("The metric does not exist")


		ensemble_mean = list()
		model_names= list()		
		models_numbers = np.arange(n_models)
		for k in range(kcross):
		    np.random.shuffle(models_numbers)
		    sub_models_numbers=models_numbers[0:Nnet]
		    base_model_name = model_fold+"/model_"
		 
		    rows = list()
		    subject_ids = list()
		    for base_case_folder in glob(folder+'/*'):

			        subject_id= os.path.basename(base_case_folder)
			        subject_ids.append(subject_id)
			        case_folder=os.path.join(base_case_folder,base_model_name)
			        
			            
			        if metric=='dice':
			            hard_dice = [dice_coefficient(load_segmentation(gt_directory,subject_id,True), ensemble_segmentacion(case_folder,sub_models_numbers)) ]
			            rows.append(hard_dice)
			        elif metric=='brier_plus': 
			             brier= [brier_plus(case_folder,subject_id,ensemble=True,mfiles=sub_models_numbers) ]   
			             print('Brier Score: ',brier)
			             rows.append(brier)   
			        elif metric=='brier':     
			             brier= [brier_score(case_folder,subject_id,ensemble=True,mfiles=sub_models_numbers) ]   
			             print('Brier Score: ',brier)
			             rows.append(brier) 
			        elif metric=='variance':     
			             var= [get_ensemble_variance(case_folder,mfiles=sub_models_numbers) ]   
			             print('Variance: ',var)
			             rows.append(var)    
			        elif metric=='segment_variance':     
			             var= [get_ensemble_variance(case_folder,mfiles=sub_models_numbers,segment=True) ]   
			             print('Variance: ',var)
			             rows.append(var) 
			        elif metric=='save_ensemble_predictions':
			             save_ensemble_pred(case_folder,model_fold,subject_id,mfiles=sub_models_numbers)
                            

		    model_name=metric+'_ensemble_'+str(Nnet)+'_cross'+str(k)
		    model_names.append(model_name)
		    ensure_dir(os.path.join('./',save_results_fold,model_fold)    )
		    df = pd.DataFrame.from_records(rows, columns=header, index=subject_ids)
		    df.to_csv(os.path.join('./',save_results_fold,model_fold,model_name+'.csv')) 
		    ensemble_mean.append(df.mean())       
		df_means = pd.DataFrame.from_records(ensemble_mean, columns=header, index=model_names)
		df_means.to_csv(os.path.join('./',save_results_fold,model_fold,'mean_'+metric+'_ensemble_'+str(Nnet)+'.csv'))   

				              



if __name__ == "__main__":


	parser = ConfigParser()
	parser.read('config_file.ini')

	gt_directory=parser["DEFAULT"].get('image_source_dir')
	folder=	parser["DEFAULT"].get("segmentation_directory")
	save_results_fold=parser["DEFAULT"].get("results_directory")
	model_folds=parser["ENSEMBLE"].get("pretrained_models_folds").split(",")	
	n_models= parser["ENSEMBLE"].getint("n_models")

	metrics=parser["TEST"].get("metrics").split(",")
	Nnets=parser["TEST"].get('Nnet').split(',')
	kcross=parser["TEST"].getint('kcross')

	for model_fold in model_folds:

			testing_models(metrics,model_fold,n_models,folder,save_results_fold)
			for Nnet in Nnets:
				testing_ensemble(int(Nnet),kcross,metrics,model_fold,n_models,folder,save_results_fold)
			




