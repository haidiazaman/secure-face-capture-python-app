import numpy as np
    
# def adjust_bbox(face_bbox,multiplier): # direct adjustment of the face_bbox of the crop using the multiplier
#     l,t,r,b=face_bbox
#     original_face_height,original_face_width=b-t,r-l
#     original_centre_x,original_centre_y=(r+l)/2,(b+t)/2
#     new_face_height,new_face_width=original_face_height*multiplier,original_face_width*multiplier
#     new_l,new_r=original_centre_x-new_face_width/2,original_centre_x+new_face_width/2
#     new_t,new_b=original_centre_y-new_face_height/2,original_centre_y+new_face_height/2
#     bbox=[new_l,new_t,new_r,new_b]
#     bbox=[int(x) for x in bbox]
#     return bbox
def adjust_bbox(image_shape,face_bbox, multiplier):
    full_image_height,full_image_width = image_shape[0],image_shape[1]
    l, t, r, b = face_bbox
    original_face_height, original_face_width = b - t, r - l
    original_centre_x, original_centre_y = (r + l) / 2, (b + t) / 2
    new_face_height, new_face_width = original_face_height * multiplier, original_face_width * multiplier
    
    # Ensure non-negative bounding box coordinates
    new_l = max(0, original_centre_x - new_face_width / 2)
    new_r = min(full_image_width, original_centre_x + new_face_width / 2)
    new_t = max(0, original_centre_y - new_face_height / 2)
    new_b = min(full_image_height ,original_centre_y + new_face_height / 2)

    bbox = [new_l, new_t, new_r, new_b]
    bbox = [int(x) for x in bbox]
    return bbox

def load_image(input_params):
    image_path=input_params['image_path']
    bbox=input_params['bbox']
    resize_img_dim=input_params['resize_img_dim']
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    l, t, r, b = bbox
    l, r = max(0, l), min(image.shape[1] - 1, r)
    t, b = max(0, t), min(image.shape[0] - 1, b)
    l,t,r,b=int(l),int(t),int(r),int(b)
    image = cv2.resize(image[t:b, l:r], (resize_img_dim,resize_img_dim))
    return image

def balance_dataset(df, label_column):  # to make function more generic use *cols then iterate thru all col in this
    unique_labels = df[label_column].unique()
    class_counts = {label: len(df[df[label_column] == label]) for label in unique_labels}
    max_count = max(class_counts.values())
    balanced_dfs = []
    for label in unique_labels: # Oversample the minority classes
        class_df = df[df[label_column] == label]
        ratio = max_count / class_counts[label] # Calculate the oversampling ratio
        ratio = max(1,round(ratio)) 
        for _ in range(ratio):
            balanced_dfs.append(class_df)
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    return balanced_df


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def convert_seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return int(hours), int(minutes), int(seconds)

# accuracy function
def get_accuracy(outputs,labels):
    predicted_classes = torch.argmax(nn.Softmax(dim=1)(outputs),dim=1).cpu().detach().numpy()
    correct_predictions=(predicted_classes==labels.cpu().detach().numpy()).sum().item()
    accuracy = correct_predictions/labels.size(0) * 100
    return accuracy


##############################
# functions for case study #
##############################

import numpy as np
from sklearn.metrics import confusion_matrix

def get_confusion_matrix_and_indices(gt, pred, num_classes):
    """
    Calculate the confusion matrix and obtain indices of images for each cell in the matrix.

    Parameters:
    - gt: Ground truth labels
    - pred: Predicted labels
    - num_classes: Number of classes

    Returns:
    - cm: Confusion matrix
    - cm_indices: Indices of images for each cell in the confusion matrix
    """
    gt, pred = np.array(gt), np.array(pred)
    
    # Get indices of each cell in the confusion matrix
    cm_indices = np.array([[np.where((gt == i) & (pred == j))[0] for j in range(num_classes)] for i in range(num_classes)], dtype=object)
    
    # Calculate the confusion matrix
    cm = confusion_matrix(gt, pred)
    
    return cm, cm_indices


import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_case_study(cm_indices, class_label, class_predicted, class_mapping, softmax_result, num_images_to_plot, n_cols, counter=0, highlight_threshold=0.9):
    """
    Plot images for a given class label and predicted class.

    Parameters:
    - cm_indices: Confusion matrix indices
    - class_label: Ground truth class label
    - class_predicted: Predicted class label
    - class_mapping: Mapping of class labels to names
    - softmax_result: Softmax scores for predictions
    - num_images_to_plot: Number of images to plot
    - n_cols: Number of columns in the plot grid
    - counter: Counter for indexing images
    - highlight_threshold: Threshold for highlighting images based on softmax scores
    """
    indices = cm_indices[class_label, class_predicted]
    print(f"Number of indices for (gt: {class_mapping[class_label]}, p: {class_mapping[class_predicted]}) is: {len(indices)}\n")
    
    indices = indices[:num_images_to_plot]
    n_rows = int(np.ceil(num_images_to_plot / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    
    for i, index in tqdm(enumerate(indices)):
        softmax_score = [round(x, 3) for x in softmax_result[index]]
        title = f"{index}\n{softmax_score}"
        image = test_dataset[index][0].permute(1, 2, 0).numpy() 
        
        ax = axes.flatten()[i]
        ax.imshow(image)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(title, fontsize=9)
    
    plt.subplots_adjust(hspace=0.1, wspace=0.05)
    plt.gcf().set_size_inches(2.4 * n_cols, 3.0 * n_rows)
    plt.show()

def plot_softmax_distribution_per_class(cm_indices, class_label, class_predicted, class_mapping, num_classes, softmax_result):
    """
    Plot the distribution of softmax scores for a specific pair of ground truth and predicted classes.

    Parameters:
    - cm_indices: Confusion matrix indices
    - class_label: Ground truth class label
    - class_predicted: Predicted class label
    - class_mapping: Mapping of class labels to names
    - num_classes: Number of classes
    - softmax_result: Softmax scores for predictions
    """
    current_softmax_result = np.array([softmax_result[index] for index in tqdm(cm_indices[class_label, class_predicted])])
    
    fig, axes = plt.subplots(1, num_classes, figsize=(5 * num_classes, 5))
    
    for index in range(num_classes):
        # Extract scores for the current index
        scores = current_softmax_result[:, index]
        mean_score = round(np.mean(scores), 4)
        median_score = round(np.median(scores), 4)

        # Create a histogram for the current index
        axes[index].hist(scores, bins=10, color='blue', alpha=0.7)
        axes[index].set_xlabel('Scores')
        axes[index].set_ylabel('Frequency')
        axes[index].set_title(f'Softmax score distribution for index {class_mapping[index]}\nMean: {mean_score}, Median: {median_score}')
        axes[index].grid(True)

    # Adjust spacing between subplots
    plt.suptitle(f'GT: {class_mapping[class_label]}, P: {class_mapping[class_predicted]}')
    plt.tight_layout()
    
    # Show the subplots
    plt.show()